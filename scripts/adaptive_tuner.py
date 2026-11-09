#!/usr/bin/env python3
"""Adaptive OmniCortex scale tuning controller and local/remote worker client.

The controller samples a broad scale range, evaluates a batch, narrows around
its best observed score, and repeats until either the requested autotuning
resolution or the statistical margin-of-error criterion is reached.

Remote workers expose only control metadata/results. Model bytes remain on the
worker unless explicitly transferred later with the Model Transfer service.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def wilson_interval(correct:int,total:int,z:float=1.96):
    if total<=0: raise ValueError("total must be > 0")
    p=correct/total; den=1+z*z/total
    c=(p+z*z/(2*total))/den
    h=z*math.sqrt((p*(1-p)+z*z/(4*total))/total)/den
    return max(0,c-h), min(1,c+h)


def parse_eval(obj:dict[str,Any]):
    if 'score' in obj: score=float(obj['score'])
    elif 'correct' in obj and 'total' in obj: score=int(obj['correct'])/max(1,int(obj['total']))
    else: raise ValueError('result needs score or correct+total')
    if not 0<=score<=1: raise ValueError('score out of range')
    out=dict(obj); out['score']=score
    if 'correct' in out and 'total' in out:
        out['lower_ci'],out['upper_ci']=wilson_interval(int(out['correct']),int(out['total']))
    return out

@dataclass
class CandidateResult:
    scale: float
    status: str
    score: float|None=None
    lower_ci: float|None=None
    upper_ci: float|None=None
    correct: int|None=None
    total: int|None=None
    worker: str|None=None
    model: str|None=None
    error: str|None=None

class WorkerClient:
    def __init__(self, base_url:str, token:str='', timeout:int=120, cafile:str=''):
        self.base=base_url.rstrip('/')
        self.token=token
        self._expect={}
        self.timeout=timeout
        self.ssl=None
        if self.base.startswith('https://'):
            import oc_security
            self.ssl=oc_security.client_ssl_context(cafile)
    def _request(self,method,path,payload=None):
        data=None
        if payload is not None:
            data=json.dumps(payload).encode();
        req=urllib.request.Request(self.base+path,data=data,method=method,headers={'Content-Type':'application/json','Authorization':f'Bearer {self.token}'} if self.token else {'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=self.timeout,context=self.ssl) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body=e.read().decode(errors='replace')
            raise RuntimeError(f'worker HTTP {e.code}: {body}') from e
    def health(self): return self._request('GET','/health')
    def submit(self,job):
        # A fresh nonce per job; the worker echoes it inside every signed result.
        if self.token and 'nonce' not in job:
            import secrets
            job={**job,'nonce':secrets.token_urlsafe(24)}
        if job.get('job_id') and job.get('nonce'):
            self._expect[job['job_id']]=(job['nonce'],float(job['scale']))
        r=self._request('POST','/api/jobs',job)
        if self.token:
            self._check(r)
            self._expect.setdefault(r['job_id'],(job.get('nonce'),float(job['scale'])))
        return r
    def get(self,job_id):
        r=self._request('GET',f'/api/jobs/{job_id}')
        if self.token:
            self._check(r)
            if r.get('job_id')!=job_id:
                raise RuntimeError(f"worker answered a request for {job_id!r} with job {r.get('job_id')!r} (replayed?)")
        return r
    def _check(self,r):
        """Refuse results that are unsigned, altered, or answer a different request."""
        import oc_security
        if not oc_security.verify_result(self.token,r):
            raise RuntimeError(f"worker result for {r.get('job_id')!r} has a missing or invalid signature; refusing it")
        want=self._expect.get(r.get('job_id'))
        if want and (r.get('nonce')!=want[0] or abs(float(r.get('scale',0))-want[1])>1e-12):
            raise RuntimeError(f"worker result for {r.get('job_id')!r} does not match the submitted job (replayed?)")
    def delete_model(self, path): return self._request('POST','/api/models/delete',{'path':path})
    def wait(self,job_id,poll=1.0):
        while True:
            j=self.get(job_id)
            if j.get('status') in {'completed','failed','cancelled','deleted_underperformer'}: return j
            time.sleep(poll)

class LocalWorkerClient(WorkerClient):
    def __init__(self, source_model:str, profile:str, suppressor:str, output_dir:str, evaluator_cmd:str='', scratch_policy:str='auto'):
        self.source_model=os.path.abspath(source_model); self.profile=os.path.abspath(profile); self.suppressor=os.path.abspath(suppressor); self.output_dir=os.path.abspath(output_dir); self.evaluator_cmd=evaluator_cmd; self.scratch_policy=scratch_policy
    def submit(self,job):
        scale=float(job['scale']); tag=f"{int(round(scale*1000)):03d}"; kind='supp' if scale<1 else ('amp' if scale>1 else 'base100')
        out=os.path.join(self.output_dir,f"{Path(self.source_model).stem}-{kind}{tag}.gguf")
        cmd=[os.environ.get('PYTHON','python3'),self.suppressor,'--gguf',self.source_model,'--h_neurons',self.profile,'--scale',str(scale),'--out',out]
        try:
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=int(job.get('timeout',3600)))
            if p.returncode!=0: raise RuntimeError((p.stderr or p.stdout)[-4000:])
            result={'status':'completed','scale':scale,'model':out,'worker':'local'}
            if self.evaluator_cmd:
                env=os.environ.copy(); env.update({'OC_MODEL':out,'OC_SCALE':str(scale),'OC_PROFILE':self.profile})
                ecmd=self.evaluator_cmd.format(model=out,scale=scale,profile=shlex.quote(self.profile))
                ep=subprocess.run(ecmd,shell=True,capture_output=True,text=True,timeout=int(job.get('timeout',3600)),env=env)
                if ep.returncode!=0: raise RuntimeError((ep.stderr or ep.stdout)[-4000:])
                text=ep.stdout.strip(); obj=None
                for i in range(len(text)-1,-1,-1):
                    if text[i]=='{':
                        try: obj=json.loads(text[i:]); break
                        except: pass
                if obj is None: raise RuntimeError('evaluator returned no JSON')
                result.update(parse_eval(obj))
            return result
        except Exception as e:
            return {'status':'failed','scale':scale,'worker':'local','error':str(e)}
    def get(self,job_id): raise NotImplementedError
    def delete_model(self, path):
        p=os.path.abspath(path)
        try:
            os.remove(p)
            return {'ok': True}
        except FileNotFoundError:
            return {'ok': True, 'missing': True}
    def wait(self,job_id,poll=0): raise NotImplementedError

class AdaptiveTuner:
    def __init__(self, state_path: str):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    def _load(self):
        if self.state_path.is_file():
            return json.loads(self.state_path.read_text())
        return {"version": 2, "batches": [], "results": [], "status": "new"}

    def _save(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def _record(self, result):
        self.state["results"].append(result)
        self._save()

    @staticmethod
    def _sample_interval(lo: float, hi: float, n: int, interior: bool = False) -> list[float]:
        if n <= 1:
            return [round((lo + hi) / 2.0, 10)]
        if interior:
            width = hi - lo
            return [round(lo + ((i + 1) / (n + 1)) * width, 10) for i in range(n)]
        if n == 2:
            return [round(lo, 10), round(hi, 10)]
        step = (hi - lo) / (n - 1)
        return [round(lo + i * step, 10) for i in range(n)]

    @staticmethod
    def _narrow_interval(batch_results, lo, hi):
        good = sorted(
            (r for r in batch_results if r.get("status") == "completed" and r.get("score") is not None),
            key=lambda r: r["scale"],
        )
        if not good:
            return lo, hi
        best = max(good, key=lambda r: float(r["score"]))
        idx = good.index(best)
        b = float(best["scale"])
        if len(good) == 1:
            width = max((hi - lo) / 4.0, 1e-6)
            return max(lo, b - width), min(hi, b + width)
        left = float(good[idx - 1]["scale"]) if idx > 0 else lo
        right = float(good[idx + 1]["scale"]) if idx + 1 < len(good) else hi
        # Contract halfway toward the best point on each side. This ensures the
        # next interval is strictly narrower even when the best point was an
        # endpoint of the sampled batch.
        if idx == 0:
            new_hi = b + 0.5 * (right - b)
            return b, min(hi, new_hi)
        if idx == len(good) - 1:
            new_lo = b - 0.5 * (b - left)
            return max(lo, new_lo), b
        new_lo = b - 0.5 * (b - left)
        new_hi = b + 0.5 * (right - b)
        return max(lo, new_lo), min(hi, new_hi)

    def run(
        self,
        *,
        client,
        minimum: float,
        maximum: float,
        initial_step: float,
        autotune_resolution: float,
        batch_size: int,
        margin_of_error: float,
        baseline_score: float | None = None,
        max_batches: int = 20,
        timeout: int = 3600,
        auto_delete: bool = False,
    ):
        if not (minimum < maximum):
            raise ValueError("minimum must be < maximum")
        if initial_step <= 0 or autotune_resolution <= 0:
            raise ValueError("steps must be > 0")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if margin_of_error < 0:
            raise ValueError("margin_of_error must be >= 0")

        self.state["status"] = "running"
        self.state.setdefault("results", [])
        self.state.setdefault("batches", [])
        self.state["config"] = {
            "minimum": minimum,
            "maximum": maximum,
            "initial_step": initial_step,
            "autotune_resolution": autotune_resolution,
            "batch_size": batch_size,
            "margin_of_error": margin_of_error,
            "baseline_score": baseline_score,
            "auto_delete": auto_delete,
        }
        self._save()

        if self.state.get("next_interval"):
            lo, hi = self.state["next_interval"]
        else:
            lo, hi = minimum, maximum
        seen = {round(float(r["scale"]), 10) for r in self.state["results"] if "scale" in r}

        for batch_number in range(1, max_batches + 1):
            if hi - lo <= autotune_resolution:
                self.state["status"] = "resolution_reached"
                self._save()
                return self.state

            points = self._sample_interval(lo, hi, batch_size, interior=bool(self.state["batches"]))
            if any(round(p, 10) in seen for p in points):
                points = self._sample_interval(lo, hi, batch_size, interior=True)
            points = [p for p in points if round(p, 10) not in seen]
            if not points:
                self.state["status"] = "no_new_points"
                self._save()
                return self.state

            batch = {"batch": batch_number, "lo": lo, "hi": hi, "scales": points}
            self.state["batches"].append(batch)
            self._save()

            batch_results = []
            for scale in points:
                job_id = str(uuid.uuid4())
                job = {"job_id": job_id, "scale": scale, "timeout": timeout}
                try:
                    submitted = client.submit(job)
                    if submitted.get("status") in {"completed", "failed", "cancelled", "deleted_underperformer"}:
                        final = submitted
                    else:
                        final = client.wait(submitted.get("job_id", job_id))
                except Exception as exc:
                    final = {"status": "failed", "scale": scale, "worker": getattr(client, "base", "remote"), "error": str(exc)}
                final["scale"] = scale
                self._record(final)
                batch_results.append(final)
                seen.add(round(scale, 10))

            valid = [r for r in batch_results if r.get("status") == "completed" and r.get("score") is not None]
            if not valid:
                self.state["status"] = "batch_failed"
                self._save()
                return self.state

            best = max(valid, key=lambda r: float(r["score"]))
            if baseline_score is not None:
                for r in valid:
                    if r.get("model") and r["model"] != best.get("model"):
                        upper = r.get("upper_ci")
                        below = (upper < baseline_score - margin_of_error) if upper is not None else (float(r["score"]) < baseline_score - margin_of_error)
                        if auto_delete and below:
                            try:
                                client.delete_model(r["model"])
                                r["status"] = "deleted_underperformer"
                            except Exception as exc:
                                r["delete_error"] = str(exc)

            if best.get("lower_ci") is not None and best.get("upper_ci") is not None:
                ci_width = float(best["upper_ci"]) - float(best["lower_ci"])
                if ci_width <= margin_of_error:
                    self.state["status"] = "margin_reached"
                    self.state["best"] = best
                    self._save()
                    return self.state

            lo, hi = self._narrow_interval(valid, lo, hi)
            self.state["next_interval"] = [lo, hi]
            self._save()
            if hi - lo <= autotune_resolution:
                self.state["status"] = "resolution_reached"
                self.state["best"] = best
                self._save()
                return self.state

            # Use the configured initial step as the upper bound on interval width;
            # subsequent batches are governed by the actual tested points.
            if (hi - lo) > initial_step * batch_size:
                center = float(best["scale"])
                half = initial_step * batch_size / 2
                lo, hi = max(minimum, center - half), min(maximum, center + half)

        self.state["status"] = "max_batches_reached"
        valid = [r for r in self.state["results"] if r.get("status") == "completed" and r.get("score") is not None]
        if valid:
            self.state["best"] = max(valid, key=lambda r: float(r["score"]))
        self._save()
        return self.state

def main(argv=None):
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    s=sub.add_parser('tune'); s.add_argument('--state',required=True); s.add_argument('--worker',required=True); s.add_argument('--min',type=float,default=.1); s.add_argument('--max',type=float,default=.9); s.add_argument('--initial-step',type=float,default=.1); s.add_argument('--resolution',type=float,default=.01); s.add_argument('--batch-size',type=int,default=2); s.add_argument('--margin-of-error',type=float,default=.03); s.add_argument('--baseline-score',type=float); s.add_argument('--max-batches',type=int,default=20); s.add_argument('--auto-delete',action='store_true',help='delete candidates that underperform the best by more than the margin'); s.add_argument('--token',default='',help='discouraged: prefer --token-file or $OC_TRANSFER_TOKEN'); s.add_argument('--token-file',default=''); s.add_argument('--cafile',default='')
    # --worker local: write candidates on this machine (where the transfer watcher can pick them up) and
    # score each with --evaluator, e.g. scripts/eval_lmstudio.py for LM Studio on another machine.
    s.add_argument('--source',help='local worker: the GGUF to scale'); s.add_argument('--profile',help='local worker: h_neurons.json')
    s.add_argument('--output-dir',help='local worker: where candidates go (the transfer watcher\'s folder)')
    s.add_argument('--suppressor',default=str(Path(__file__).resolve().parent/'suppress_gguf.py'))
    s.add_argument('--evaluator',default='',help='local worker: command printing a JSON score; {model} {scale} {profile}')
    a=p.parse_args(argv)
    if a.cmd=='tune':
        import oc_security
        if a.worker=='local':
            missing=[f for f in ('source','profile','output_dir') if not getattr(a,f)]
            if missing: p.error('--worker local needs '+', '.join('--'+m.replace('_','-') for m in missing))
            if not a.evaluator: p.error('--worker local needs --evaluator (e.g. scripts/eval_lmstudio.py)')
            os.makedirs(a.output_dir,exist_ok=True)
            client=LocalWorkerClient(a.source,a.profile,a.suppressor,a.output_dir,evaluator_cmd=a.evaluator)
        else:
            client=WorkerClient(a.worker, token=oc_security.resolve_token(a.token,a.token_file), cafile=a.cafile)
        print(json.dumps(AdaptiveTuner(a.state).run(client=client,minimum=a.min,maximum=a.max,initial_step=a.initial_step,autotune_resolution=a.resolution,batch_size=a.batch_size,margin_of_error=a.margin_of_error,baseline_score=a.baseline_score,max_batches=a.max_batches,auto_delete=a.auto_delete),indent=2)); return 0
    return 0
if __name__=='__main__': raise SystemExit(main())
