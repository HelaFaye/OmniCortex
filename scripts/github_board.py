#!/usr/bin/env python3
"""
A scoreboard kept in a GitHub repository (docs/SHARING.md#a-scoreboard-on-github): no server to host.

- **Submitting.** A score card arrives as an issue. The card itself is public JSON. Its verification
  log (each item's reply) is sealed to the board's public key (X25519, HKDF-SHA256,
  ChaCha20-Poly1305), so only the board's own workflow can read it. A long log continues in comments.
- **Accepting.** The board's workflow runs `github_board.py action`. It unseals the log with the
  BOARD_KEY secret, regrades every reply with the TestQA bank (scoreboard_server.verify_card), and then:
  - a verified card is committed as cards/<sha[:2]>/<sha>/<variant>/<issue>.json, and board.json is rebuilt;
  - a rejected or duplicate card is not committed;
  - either way, the workflow comments with the outcome and closes the issue.
  The log is never committed.
- **Reading.** board.json holds every model and variant's aggregate, the speed measurements cards carry,
  the fitted speed prediction weights (perf_predict.py), and the measurement weights
  (measure_weights.py). Studio and anyone else read it from raw.githubusercontent.com.

    python scripts/github_board.py init ./my-board --neuronscope-repo owner/neuronscope   # skeleton + keys
    python scripts/github_board.py submit --board owner/my-board --card c.json --log l.json  # (Studio does this)
    python scripts/github_board.py rebuild ./my-board
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

MARK = "<!-- neuronscope-card v1 -->"
INFO = b"neuronscope-board-v1"
PART = 60000                         # GitHub issue bodies and comments hold 65536 characters
CARD_MAX = 50000
MAX_PARTS = 40
API = "https://api.github.com"
REPO_RE = re.compile(r"[\w.-]+/[\w.-]+")


# ================================================================ sealing

def keygen() -> tuple[str, str]:
    """-> (private, public), both base64 raw X25519 keys."""
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    from cryptography.hazmat.primitives import serialization as ser
    k = X25519PrivateKey.generate()
    raw = dict(encoding=ser.Encoding.Raw)
    priv = k.private_bytes(format=ser.PrivateFormat.Raw, encryption_algorithm=ser.NoEncryption(), **raw)
    pub = k.public_key().public_bytes(format=ser.PublicFormat.Raw, **raw)
    return base64.b64encode(priv).decode(), base64.b64encode(pub).decode()


def _key(shared: bytes, eph: bytes, pub: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=eph + pub, info=INFO).derive(shared)


def seal(data: bytes, pub_b64: str) -> str:
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    from cryptography.hazmat.primitives import serialization as ser
    pub = base64.b64decode(pub_b64)
    eph = X25519PrivateKey.generate()
    ephp = eph.public_key().public_bytes(encoding=ser.Encoding.Raw, format=ser.PublicFormat.Raw)
    key = _key(eph.exchange(X25519PublicKey.from_public_bytes(pub)), ephp, pub)
    nonce = os.urandom(12)
    ct = ChaCha20Poly1305(key).encrypt(nonce, zlib.compress(data, 9), INFO)
    return base64.b64encode(ephp + nonce + ct).decode()


def unseal(blob: str, priv_b64: str) -> bytes:
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    from cryptography.hazmat.primitives import serialization as ser
    raw = base64.b64decode(re.sub(r"\s+", "", blob))
    ephp, nonce, ct = raw[:32], raw[32:44], raw[44:]
    k = X25519PrivateKey.from_private_bytes(base64.b64decode(priv_b64))
    pub = k.public_key().public_bytes(encoding=ser.Encoding.Raw, format=ser.PublicFormat.Raw)
    key = _key(k.exchange(X25519PublicKey.from_public_bytes(ephp)), ephp, pub)
    return zlib.decompress(ChaCha20Poly1305(key).decrypt(nonce, ct, INFO), bufsize=2**20)


# ================================================================ issues

def issue_parts(card: dict, log: dict, pub_b64: str) -> tuple[str, str, list[str]]:
    """-> (title, body, comments). The sealed log is split over the body and comments when long."""
    sealed = seal(json.dumps(log, separators=(",", ":")).encode(), pub_b64)
    chunks = [sealed[i:i + PART] for i in range(0, len(sealed), PART)] or [""]
    if len(chunks) > MAX_PARTS:
        raise ValueError(f"the verification log is too large ({len(sealed)} characters sealed)")
    m = card.get("model") or {}
    title = f"card: {m.get('file', '?')} ({card.get('variant_id', '?')})"
    cj = json.dumps(card, indent=1)
    if len(cj) > CARD_MAX:
        raise ValueError("the card is too large for an issue")
    body = (f"{MARK}\nA OmniCortex score card. This board's workflow verifies it, then commits it or says why "
            f"not.\n\n```json\n{cj}\n```\n\nVerification log, sealed to the board's key "
            f"(part 1 of {len(chunks)}):\n\n```sealed\n{chunks[0]}\n```\n")
    comments = [f"{MARK} part {i + 2} of {len(chunks)}\n\n```sealed\n{c}\n```\n" for i, c in enumerate(chunks[1:])]
    return title, body, comments


def parse_issue(body: str, comments: list[str] = ()) -> tuple[dict, str]:
    if MARK not in (body or ""):
        raise ValueError("not a OmniCortex card")
    m = re.search(r"```json\n(.*?)\n```", body, re.S)
    if not m:
        raise ValueError("no card JSON in the issue")
    card = json.loads(m.group(1))
    total = int((re.search(r"\(part 1 of (\d+)\)", body) or [0, 1])[1])
    parts = {1: (re.search(r"```sealed\n(.*?)\n```", body, re.S) or [None, ""])[1]}
    for c in comments:
        pm = re.search(re.escape(MARK) + r" part (\d+) of (\d+)", c or "")
        if pm and int(pm.group(2)) == total:
            parts[int(pm.group(1))] = (re.search(r"```sealed\n(.*?)\n```", c, re.S) or [None, ""])[1]
    if sorted(parts) != list(range(1, total + 1)):
        raise ValueError(f"the sealed log has {len(parts)} of {total} parts")
    return card, "".join(parts[i] for i in range(1, total + 1))


def _gh(method: str, path: str, token: str | None, body=None, timeout: float = 30):
    req = urllib.request.Request(API + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Accept": "application/vnd.github+json", "User-Agent": "neuronscope",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def board_key(repo: str, branch: str = "main") -> str:
    with urllib.request.urlopen(f"https://raw.githubusercontent.com/{repo}/{branch}/board.pub", timeout=20) as r:
        k = r.read().decode().strip()
    if len(base64.b64decode(k)) != 32:
        raise ValueError(f"{repo}/board.pub is not an X25519 public key")
    return k


def submit(repo: str, card: dict, log: dict, token: str, pub: str | None = None) -> dict:
    """Open the issue (and its comments). -> {"status": "submitted", "url", "number"}."""
    if not REPO_RE.fullmatch(repo):
        raise ValueError("a board is owner/repo")
    if card.get("visibility") != "public":
        raise ValueError("everything on a GitHub board is public; this card is private (an unlisted model). "
                         "List the model by hand to submit it, or use a scoreboard server")
    if not token:
        raise ValueError("submitting to GitHub needs a token that may open issues (Settings → Sharing results)")
    title, body, comments = issue_parts(card, log, pub or board_key(repo))
    iss = _gh("POST", f"/repos/{repo}/issues", token, {"title": title, "body": body})
    for c in comments:
        _gh("POST", f"/repos/{repo}/issues/{iss['number']}/comments", token, {"body": c})
    return {"status": "submitted", "note": "the board's workflow verifies it and answers on the issue",
            "url": iss.get("html_url"), "number": iss.get("number")}


# ================================================================ the board's side

def card_path(board: Path, card: dict, number: int) -> Path:
    sha = card["model"]["sha256"]
    return board / "cards" / sha[:2] / sha / str(card.get("variant_id") or "none") / f"{int(number)}.json"


def all_cards(board: Path) -> list[dict]:
    out = []
    for p in sorted((board / "cards").rglob("*.json")) if (board / "cards").is_dir() else []:
        try:
            out.append(json.loads(p.read_text()))
        except ValueError:
            continue
    return out


def rebuild(board: Path) -> dict:
    import scoreboard_server as sb
    cards = all_cards(board)
    rows = sb.aggregate(cards)
    out = {"updated": int(time.time()), "cards": len(cards), "rows": rows}
    try:
        import perf_predict
        samples = perf_predict.samples_from_board(rows)
        if samples:
            out["perf_model"] = perf_predict.fit(samples).to_json()
    except Exception as e:                     # a board still publishes without the extras
        out["perf_model_error"] = str(e)[:200]
    try:
        import measure_weights
        out["measure_weights"] = measure_weights.from_board(rows)
    except Exception as e:
        out["measure_weights_error"] = str(e)[:200]
    (board / "board.json").write_text(json.dumps(out, indent=1))
    return out


def accept(board: Path, number: int, body: str, comments: list[str], priv: str, tasks=None) -> dict:
    """Verify one issue's card; commit-ready files on success. -> {"status", "note", "path"?}."""
    import scoreboard_server as sb
    import testqa as tq
    try:
        card, sealed = parse_issue(body, comments)
        log = json.loads(unseal(sealed, priv))
    except Exception as e:
        return {"status": "rejected", "note": f"unreadable: {type(e).__name__}: {e}"[:300]}
    if card.get("card_version") != 1 or len(str((card.get("model") or {}).get("sha256", ""))) != 64:
        return {"status": "rejected", "note": "not a version 1 card with a model SHA-256"}
    if card.get("visibility") != "public":
        return {"status": "rejected", "note": "private cards are not accepted on a public board"}
    digest = hashlib.sha256(json.dumps(log, sort_keys=True).encode()).hexdigest()
    seen = {c.get("_log_sha256") for c in all_cards(board)}
    if digest in seen:
        return {"status": "rejected", "note": "this exact result was already accepted"}
    if tasks is None:
        tasks = {t["id"]: t for t in tq.load_tasks([str(ROOT / "qa" / "bank")], None, None, 0)}
    status, note = sb.verify_card(tasks, card, log)
    if status != "verified":
        return {"status": status, "note": note}
    p = card_path(board, card, number)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**card, "_log_sha256": digest, "_issue": int(number)}, indent=1))
    rebuild(board)
    return {"status": "verified", "note": note, "path": str(p.relative_to(board))}


def action(board: Path, number: int, out: Path) -> int:
    """In the board's workflow: read the issue through the API, accept or reject, write the outcome."""
    repo, token, priv = os.environ["GITHUB_REPOSITORY"], os.environ.get("GITHUB_TOKEN"), os.environ.get("BOARD_KEY")
    if not priv:
        res = {"status": "rejected", "note": "the board has no BOARD_KEY secret"}
    else:
        iss = _gh("GET", f"/repos/{repo}/issues/{number}", token)
        comments, page = [], 1
        while True:
            got = _gh("GET", f"/repos/{repo}/issues/{number}/comments?per_page=100&page={page}", token)
            comments += [c.get("body") or "" for c in got]
            if len(got) < 100 or page >= 5:
                break
            page += 1
        res = accept(board, number, iss.get("body") or "", comments, priv)
    out.write_text(json.dumps(res))
    print(json.dumps(res))
    return 0


WORKFLOW = """name: accept score cards
on:
  issues:
    types: [opened, edited]
  issue_comment:
    types: [created]
permissions:
  contents: write
  issues: write
concurrency: board
jobs:
  accept:
    if: contains(github.event.issue.body, '<!-- neuronscope-card v1 -->') && github.event.issue.state == 'open'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/checkout@v4
        with: {repository: __NS_REPO__, ref: __NS_REF__, path: ns}
      - uses: actions/setup-python@v5
        with: {python-version: '3.12'}
      - run: pip install cryptography numpy
      - name: verify
        env: {BOARD_KEY: '${{ secrets.BOARD_KEY }}', GITHUB_TOKEN: '${{ github.token }}'}
        run: python ns/scripts/github_board.py action --board . --issue ${{ github.event.issue.number }} --out /tmp/result.json
      - name: commit
        run: |
          if [ "$(python -c "import json;print(json.load(open('/tmp/result.json'))['status'])")" = verified ]; then
            git config user.name "board" && git config user.email "board@users.noreply.github.com"
            git add cards board.json && git commit -m "card from #${{ github.event.issue.number }}" && git push
          fi
      - name: answer
        env: {GH_TOKEN: '${{ github.token }}'}
        run: |
          S=$(python -c "import json;r=json.load(open('/tmp/result.json'));print(r['status']+': '+r['note'])")
          case "$S" in "rejected: the sealed log has"*) exit 0;; esac   # more parts are still coming
          gh issue comment ${{ github.event.issue.number }} --body "$S"
          gh issue close ${{ github.event.issue.number }}
"""


def init(board: Path, ns_repo: str, ns_ref: str = "main") -> dict:
    """A new board repository's files, and its key pair. The private key goes in the repo's BOARD_KEY secret
    and is written to board.key here for that; delete it once stored."""
    if not REPO_RE.fullmatch(ns_repo):
        raise ValueError("--neuronscope-repo is owner/repo")
    board.mkdir(parents=True, exist_ok=True)
    priv, pub = keygen()
    (board / "board.pub").write_text(pub + "\n")
    kp = board / "board.key"
    kp.write_text(priv + "\n")
    os.chmod(kp, 0o600)
    (board / ".gitignore").write_text("board.key\n")
    wf = board / ".github" / "workflows"
    wf.mkdir(parents=True, exist_ok=True)
    (wf / "accept.yml").write_text(WORKFLOW.replace("__NS_REPO__", ns_repo).replace("__NS_REF__", ns_ref))
    (board / "cards").mkdir(exist_ok=True)
    (board / "cards" / ".keep").write_text("")
    (board / "README.md").write_text(
        "# OmniCortex scoreboard\n\nScore cards arrive as issues and are verified by the workflow in "
        "`.github/workflows/accept.yml`: replies are regraded with the TestQA bank of "
        f"`{ns_repo}` ({ns_ref}). Accepted cards are in `cards/`; `board.json` aggregates them.\n\n"
        "Submit from OmniCortex Studio: Settings → Sharing results → scoreboard `github:OWNER/REPO`.\n")
    rebuild(board)
    return {"board": str(board), "public_key": pub, "private_key_file": str(kp),
            "next": ["create the repository on GitHub and push these files",
                     "Settings → Secrets and variables → Actions → New secret BOARD_KEY = the contents of board.key",
                     "then delete board.key"]}


def fetch(repo: str, branch: str = "main", timeout: float = 20) -> dict:
    with urllib.request.urlopen(f"https://raw.githubusercontent.com/{repo}/{branch}/board.json", timeout=timeout) as r:
        return json.loads(r.read())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init", help="a new board repository's files and keys")
    i.add_argument("board")
    i.add_argument("--neuronscope-repo", required=True, help="owner/repo whose TestQA bank verifies cards")
    i.add_argument("--neuronscope-ref", default="main")
    s = sub.add_parser("submit", help="open the issue for a card")
    s.add_argument("--board", required=True, help="owner/repo")
    s.add_argument("--card", required=True)
    s.add_argument("--log", required=True)
    s.add_argument("--token", default=os.environ.get("NS_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN"))
    a = sub.add_parser("action", help="(in the board's workflow) verify one issue")
    a.add_argument("--board", required=True)
    a.add_argument("--issue", type=int, required=True)
    a.add_argument("--out", required=True)
    r = sub.add_parser("rebuild", help="regenerate board.json from cards/")
    r.add_argument("board")
    f = sub.add_parser("fetch", help="print a board's board.json")
    f.add_argument("repo")
    args = p.parse_args(argv)
    try:
        if args.cmd == "init":
            print(json.dumps(init(Path(args.board), args.neuronscope_repo, args.neuronscope_ref), indent=1))
        elif args.cmd == "submit":
            print(json.dumps(submit(args.board, json.loads(Path(args.card).read_text()),
                                    json.loads(Path(args.log).read_text()), args.token), indent=1))
        elif args.cmd == "action":
            return action(Path(args.board), args.issue, Path(args.out))
        elif args.cmd == "rebuild":
            out = rebuild(Path(args.board))
            print(f"{out['cards']} cards, {len(out['rows'])} rows")
        else:
            print(json.dumps(fetch(args.repo), indent=1))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
