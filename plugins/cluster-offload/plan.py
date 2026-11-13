"""Place a model across this computer and connected hosts by free GPU memory (Studio's system monitor)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def devices(view: dict, proxmox: dict | None = None) -> list[dict]:
    out = []

    def add(host, rows):
        if not rows:
            return
        for g in rows[-1].get("gpus") or []:
            tot, used = g.get("vram_total_mb"), g.get("vram_used_mb")
            if tot:
                out.append({"host": host, "gpu": g.get("name") or g.get("id"), "free_gb": round((tot - (used or 0)) / 1024, 2),
                            "total_gb": round(tot / 1024, 2)})
    add("this computer", view.get("local"))
    for h, rows in (view.get("hosts") or {}).items():
        add(h, rows)
    for n in (proxmox or {}).get("nodes") or []:
        for g in n.get("gpus") or []:
            if not any(d["host"] == n["node"] for d in out):
                out.append({"host": n["node"], "gpu": g.get("name"), "free_gb": None, "total_gb": None,
                            "note": "Proxmox lists it; run an OmniCortex agent there for its free memory"})
    return out


def place(devs: list[dict], need_gb: float, headroom: float = 0.9, port: int = 50052) -> dict:
    usable = sorted((d for d in devs if d.get("free_gb")), key=lambda d: -d["free_gb"])
    if not usable:
        return {"fits": False, "why": "no GPU memory readings: open the System page once, or connect the hosts"}
    best = usable[0]
    if best["free_gb"] * headroom >= need_gb:
        return {"fits": True, "where": best, "how": "one device: load it there (its host's Studio, or a worker)"}
    chosen, total = [], 0.0
    for d in usable:
        chosen.append(d)
        total += d["free_gb"] * headroom
        if total >= need_gb:
            break
    if total < need_gb:
        return {"fits": False, "why": f"all devices together have {total:.1f} GB usable; {need_gb:.1f} GB needed",
                "devices": chosen}
    import layer_stream
    local = [d for d in chosen if d["host"] == "this computer"]
    remote = [d for d in chosen if d["host"] != "this computer"]
    hosts = list(dict.fromkeys(d["host"] for d in remote))
    # llama-server lists RPC devices first, in --rpc order, then its own: the split follows that order
    order = [d for h in hosts for d in remote if d["host"] == h] + local
    return {"fits": True, "how": "layer split across computers (llama.cpp RPC)", "devices": order,
            "rpc_servers": ",".join(f"{h}:{port}" for h in hosts),
            "tensor_split": layer_stream.split([d["free_gb"] for d in order]),
            "steps": [f"on {h}: run the job 'Layer streaming: lend this computer's memory' (rpc-server) on its trusted "
                      f"network address, port {port}" for h in hosts] +
                     ["here: set the model's RPC servers and tensor split to the values above, then load it"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-gb", type=float, required=True)
    ap.add_argument("--ctx-gb", type=float, default=1.5)
    ap.add_argument("--headroom", type=float, default=0.9)
    ap.add_argument("--rpc-port", type=int, default=50052)
    ap.add_argument("--proxmox")
    a = ap.parse_args(argv)
    import plugin_api
    view = plugin_api.call("/api/system/monitor")
    px = json.loads(Path(a.proxmox).read_text()) if a.proxmox else None
    devs = devices(view, px)
    print(json.dumps({"devices": devs, "plan": place(devs, a.model_gb + a.ctx_gb, a.headroom, a.rpc_port)}, indent=1))


if __name__ == "__main__":
    main()
