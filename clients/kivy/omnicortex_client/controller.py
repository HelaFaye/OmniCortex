"""
What the app does, without Kivy: the saved pairing, the conversation, streaming a reply on a worker thread, the
reply checks and the review lookup, plus the text the screens show for each. app.py only draws it, so everything
here is tested on its own (tests/test_kivy_client.py).

Callbacks from the worker thread go through `post`, which the app sets to Kivy's Clock so they run on the
interface thread; tests run them directly.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from . import api

RISK_LEVELS = ((0.75, "high"), (0.5, "elevated"), (0.25, "some"))     # per-token H-neuron probability -> label
RISK_COLOURS = {"high": "ff5c5c", "elevated": "ffae42", "some": "e6d75a"}

VERDICT_TEXT = {
    "knows_but_says_otherwise": "The model's own activations favour another answer over the one it gave.",
    "doubts_own_answer": "The model's activations rate its own answer as probably false.",
    "consistent": "The model's activations agree with its answer.",
    "unsure": "No clear signal either way.",
}


def risk_level(p: float) -> str:
    return next((name for cut, name in RISK_LEVELS if p >= cut), "")


def escape_markup(s: str) -> str:
    """Kivy markup treats [ ] & specially."""
    return s.replace("&", "&amp;").replace("[", "&bl;").replace("]", "&br;")


def risk_markup(trace: dict) -> str:
    """The reply with each risky token coloured by its H-neuron probability (Kivy markup)."""
    out = []
    for piece, p in zip(trace.get("pieces") or [], trace.get("prob") or []):
        t = escape_markup(str(piece))
        lvl = risk_level(float(p))
        out.append(f"[color={RISK_COLOURS[lvl]}]{t}[/color]" if lvl else t)
    return "".join(out)


def risk_summary(trace: dict) -> str:
    prob = [float(p) for p in trace.get("prob") or []]
    if not prob:
        return "No tokens were scored."
    flagged = trace.get("flagged") or [i for i, p in enumerate(prob) if p >= trace.get("threshold", 0.5)]
    peak = max(prob)
    words = f"{len(flagged)} of {len(prob)} tokens flagged" if flagged else "No token flagged"
    return f"{words}; highest risk {peak:.2f}, mean {sum(prob) / len(prob):.2f}."


def truth_summary(r: dict) -> str:
    v = r.get("verdict")
    lines = [VERDICT_TEXT.get(v, f"Verdict: {v}")]
    a = r.get("answer") or {}
    if "p_true" in a:
        lines.append(f"Its answer: rated true with p = {a['p_true']:.2f}.")
    best = r.get("best_alternative")
    if best:
        p = next((x["p_true"] for x in r.get("alternatives") or [] if x.get("text") == best), None)
        lines.append(f"Best alternative “{best}”: p = {p:.2f}." if p is not None else f"Best alternative: {best}")
    if r.get("alt_leads_at"):
        lines.append("The alternative led at " + ", ".join(map(str, r["alt_leads_at"][:6])) +
                     (" and later layers." if len(r["alt_leads_at"]) > 6 else "."))
    if r.get("stage"):
        lines.append(f"Read at {r['stage']} (probe AUROC {r.get('stage_auc', 0):.2f}).")
    lines.append("A flag means the activations and the words disagree; it does not show intent.")
    return "\n".join(lines)


def _fmt_value(s: dict) -> str:
    v, unit = s.get("value"), s.get("unit")
    if isinstance(v, (int, float)):
        if unit == "ratio":
            return f"{100 * v:.0f}%"
        if unit in ("tok/s", "tok_s"):
            return f"{v:.1f} tok/s"
        return f"{v:.3g}" + (f" {unit}" if unit and unit not in ("freq", "count") else "")
    return str(v)


def lookup_summary(rec: dict) -> str:
    """The unified review as plain lines, grouped like Studio's card."""
    if not rec.get("available"):
        bad = [s for s in rec.get("sources") or [] if not s.get("ok")]
        why = "; ".join(f"{s.get('name') or s.get('id')}: {s.get('error')}" for s in bad)
        return f"Nothing known yet about {rec.get('model', 'this model')}." + (f"\n({why})" if why else "")
    lines, group = [f"{rec.get('model')}"], None
    for s in sorted(rec.get("stats") or [], key=lambda s: str(s.get("group"))):
        if s.get("group") != group:
            group = s.get("group")
            lines.append(f"\n{str(group).capitalize()}")
        extra = ", ".join(x for x in (s.get("variant"), f"n = {s['n']}" if s.get("n") else None,
                                       "related model" if s.get("related") else None) if x)
        lines.append(f"  {s.get('label') or s.get('key')}: {_fmt_value(s)}" + (f" ({extra})" if extra else ""))
    return "\n".join(lines)


@dataclass
class Message:
    role: str
    content: str
    check: str = ""          # Kivy markup of the risk strip, once checked
    note: str = ""           # summary of the reply checks
    model: str = ""


@dataclass
class Client:
    data_dir: str | None = None
    post: object = None                       # callable(fn): run fn on the interface thread
    pairing: api.Pairing | None = None
    models: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    busy: bool = False
    _stop: threading.Event = field(default_factory=threading.Event)

    # ------------------------------------------------------------- pairing
    def load(self) -> bool:
        self.pairing = api.load(self.data_dir)
        return self.pairing is not None

    @property
    def studio(self) -> api.Studio:
        if not self.pairing:
            raise api.StudioError("not paired with a Studio yet")
        return api.Studio(self.pairing)

    def pair(self, link: str, name: str) -> api.Pairing:
        self.pairing = api.Studio.pair(link, name.strip() or "device")
        api.save(self.pairing, self.data_dir)
        return self.pairing

    def forget(self):
        api.forget(self.data_dir)
        self.pairing, self.models, self.messages = None, [], []

    def warning(self) -> str:
        return api.Studio(self.pairing).warning if self.pairing else ""

    # ------------------------------------------------------------- models
    def refresh_models(self) -> list:
        self.models = self.studio.models()
        if self.pairing.model not in self.models and self.pairing.model != "auto":
            self.pairing.model = "auto"
        return self.models

    def choose(self, model: str):
        self.pairing.model = model or "auto"
        api.save(self.pairing, self.data_dir)

    # ------------------------------------------------------------- chat
    def _run(self, fn):
        (self.post or (lambda f: f()))(fn)

    def send(self, text: str, on_piece, on_done, on_error, background: bool = True):
        """Adds the user's message, streams the reply into a new assistant message. on_piece(msg) after each
        piece, on_done(msg) or on_error(str) at the end; all through `post`."""
        text = text.strip()
        if not text or self.busy:
            return None
        self.messages.append(Message("user", text))
        reply = Message("assistant", "", model=self.pairing.model if self.pairing else "")
        self.messages.append(reply)
        history = [{"role": m.role, "content": m.content} for m in self.messages[:-1]]
        self.busy = True
        self._stop.clear()

        def work():
            try:
                for piece in self.studio.chat_stream(history):
                    if self._stop.is_set():
                        break
                    reply.content += piece
                    self._run(lambda: on_piece(reply))
                self._run(lambda: on_done(reply))
            except api.StudioError as e:
                msg = str(e)
                self._run(lambda: on_error(msg))
            finally:
                self.busy = False

        if background:
            threading.Thread(target=work, daemon=True).start()
        else:
            work()
        return reply

    def stop(self):
        self._stop.set()

    def clear(self):
        self.messages = []

    def _context(self, reply: Message) -> list[dict]:
        i = self.messages.index(reply)
        return [{"role": m.role, "content": m.content} for m in self.messages[:i]]

    # ------------------------------------------------------------- reply checks and lookup
    def _bg(self, call, ok, err, background):
        def work():
            try:
                r = call()
                self._run(lambda: ok(r))
            except api.StudioError as e:
                msg = str(e)
                self._run(lambda: err(msg))
        if background:
            threading.Thread(target=work, daemon=True).start()
        else:
            work()

    def check(self, reply: Message, on_done, on_error, background: bool = True):
        """The H-neuron risk strip for one reply."""
        def ok(r):
            reply.check = risk_markup(r)
            reply.note = risk_summary(r)
            on_done(reply)
        self._bg(lambda: self.studio.check(self._context(reply), reply.content, reply.model or None),
                 ok, on_error, background)

    def truth(self, reply: Message, alternatives: list[str], on_done, on_error, background: bool = True):
        """Does the model's own activations agree with this reply?"""
        alts = [a.strip() for a in alternatives if a.strip()][:8]

        def ok(r):
            reply.note = truth_summary(r)
            on_done(reply)
        self._bg(lambda: self.studio.truth(self._context(reply), reply.content, alts, reply.model or None),
                 ok, on_error, background)

    def lookup(self, model: str, on_done, on_error, background: bool = True):
        model = model.strip()
        if not model:
            return on_error("type a model id or Hugging Face repo")
        self._bg(lambda: lookup_summary(self.studio.lookup(model)), on_done, on_error, background)
