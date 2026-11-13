"""Ready-made graphs for the Graphs page (docs/GRAPHS.md): start from one and change it."""
from __future__ import annotations


def _n(id, type, col, row, **params):
    return {"id": id, "type": type, "params": params, "x": 40 + col * 230, "y": 40 + row * 130}


def _e(a, ap, b, bp):
    return {"from": [a, ap], "to": [b, bp]}


TEMPLATES = {
    "screen-and-verify": {
        "name": "screen-and-verify",
        "description": "Answer a question, then screen the answer for secrets and length, check it for hallucination "
                       "from the model's own neurons, and have a judge grade it. Anything that fails goes to blocked "
                       "with the reason.",
        "nodes": [_n("q", "input", 0, 1, name="question"), _n("answer", "llm", 1, 1),
                  _n("secrets", "secrets_screen", 2, 1), _n("length", "length_screen", 3, 1, min=1, max=20000),
                  _n("neurons", "hallucination_check", 4, 1, max_risk=0.6), _n("judge", "judge", 5, 1, pass_score=7),
                  _n("ok", "output", 6, 1, name="result"),
                  _n("why", "merge", 5, 3), _n("blocked", "output", 6, 3, name="blocked")],
        "edges": [_e("q", "value", "answer", "prompt"), _e("answer", "text", "secrets", "text"),
                  _e("secrets", "pass", "length", "text"), _e("length", "pass", "neurons", "reply"),
                  _e("q", "value", "neurons", "prompt"), _e("neurons", "pass", "judge", "answer"),
                  _e("q", "value", "judge", "question"), _e("judge", "pass", "ok", "value"),
                  _e("secrets", "fail", "why", "a"), _e("neurons", "fail", "why", "b"), _e("judge", "fail", "why", "c"),
                  _e("why", "value", "blocked", "value")]},
    "retry-until-judged": {
        "name": "retry-until-judged",
        "description": "Answer, have a judge grade it, and on a low score try again with the judge's feedback, up to "
                       "three more times; then give up and say why.",
        "nodes": [_n("q", "input", 0, 1, name="question"), _n("loop", "loop", 1, 1, max=3),
                  _n("prompt", "template", 2, 1, template="{a}\n\n{b}"), _n("answer", "llm", 3, 1),
                  _n("judge", "judge", 4, 1, pass_score=7), _n("ok", "output", 5, 1, name="result"),
                  _n("fb", "template", 4, 3, template="A reviewer graded your last answer low: {a}\nAnswer the question again."),
                  _n("gave_up", "output", 5, 3, name="gave_up")],
        "edges": [_e("q", "value", "loop", "start"), _e("loop", "value", "prompt", "a"), _e("prompt", "text", "answer", "prompt"),
                  _e("answer", "text", "judge", "answer"), _e("q", "value", "judge", "question"),
                  _e("judge", "pass", "ok", "value"), _e("judge", "fail", "fb", "a"), _e("fb", "text", "loop", "again"),
                  _e("loop", "done", "gave_up", "value")]},
    "cross-model-verify": {
        "name": "cross-model-verify",
        "description": "One model answers, a second checks it; when they disagree a person decides.",
        "nodes": [_n("q", "input", 0, 1, name="question"), _n("answer", "llm", 1, 1),
                  _n("check", "cross_check", 2, 1, model=""), _n("ok", "output", 4, 1, name="result"),
                  _n("person", "approval", 3, 2, question="The second model disagrees. Accept this answer?"),
                  _n("rejected", "output", 4, 3, name="rejected")],
        "edges": [_e("q", "value", "answer", "prompt"), _e("answer", "text", "check", "answer"), _e("q", "value", "check", "question"),
                  _e("check", "pass", "ok", "value"), _e("check", "fail", "person", "value"),
                  _e("person", "approved", "ok", "value"), _e("person", "rejected", "rejected", "value")]},
    "self-consistent-answer": {
        "name": "self-consistent-answer",
        "description": "Ask five times; pass the majority answer when at least 60% agree, otherwise ask a person.",
        "nodes": [_n("q", "input", 0, 1, name="question"),
                  _n("vote", "self_consistency", 1, 1, samples=5, min_agreement=0.6), _n("ok", "output", 3, 1, name="result"),
                  _n("person", "approval", 2, 2, question="The samples disagree. Accept the most common answer?"),
                  _n("unsure", "output", 3, 3, name="unsure")],
        "edges": [_e("q", "value", "vote", "prompt"), _e("vote", "pass", "ok", "value"), _e("vote", "fail", "person", "value"),
                  _e("person", "approved", "ok", "value"), _e("person", "rejected", "unsure", "value")]},
    "code-with-tests": {
        "name": "code-with-tests",
        "description": "A worker writes code for a task; it runs against the tests in a sandbox; failures go back "
                       "to the worker, up to three more times.",
        "nodes": [_n("task", "input", 0, 1, name="task"), _n("tests", "input", 0, 3, name="tests"),
                  _n("loop", "loop", 1, 1, max=3), _n("work", "worker", 2, 1, profile="code_sandboxed"),
                  _n("run", "code_tests", 3, 1, sandbox="docker"), _n("ok", "output", 4, 1, name="result"),
                  _n("fb", "template", 3, 3, template="{b}\n\nYour last attempt failed its tests: {a}\nFix it."),
                  _n("gave_up", "output", 4, 3, name="gave_up")],
        "edges": [_e("task", "value", "loop", "start"), _e("loop", "value", "work", "task"), _e("work", "result", "run", "code"),
                  _e("tests", "value", "run", "tests"), _e("run", "pass", "ok", "value"), _e("run", "fail", "fb", "a"),
                  _e("task", "value", "fb", "b"), _e("fb", "text", "loop", "again"), _e("loop", "done", "gave_up", "value")]},
    "verify-delegated-result": {
        "name": "verify-delegated-result",
        "description": "For projects: screens a worker's result before a person reviews it. Inputs task and result; "
                       "accepted carries a result that passed, feedback says why one did not (Projects -> verification).",
        "nodes": [_n("task", "input", 0, 1, name="task"), _n("result", "input", 0, 2, name="result"),
                  _n("secrets", "secrets_screen", 1, 2), _n("length", "length_screen", 2, 2, min=20),
                  _n("judge", "judge", 3, 2, rubric="Does the result do the task completely and correctly, without "
                                                    "invented files, APIs or results?", pass_score=7),
                  _n("ok", "output", 4, 1, name="accepted"), _n("why", "merge", 4, 3), _n("fb", "output", 5, 3, name="feedback")],
        "edges": [_e("result", "value", "secrets", "text"), _e("secrets", "pass", "length", "text"),
                  _e("length", "pass", "judge", "answer"), _e("task", "value", "judge", "question"),
                  _e("judge", "pass", "ok", "value"), _e("secrets", "fail", "why", "a"), _e("length", "fail", "why", "b"),
                  _e("judge", "fail", "why", "c"), _e("why", "value", "fb", "value")]},
    "emotion-guard": {
        "name": "emotion-guard",
        "description": "Reply, then read the reply through the model's emotion vectors; a desperate or hostile peak "
                       "goes to a person before it is sent (needs the decoding lab and emotion vectors).",
        "nodes": [_n("q", "input", 0, 1, name="message"), _n("reply", "llm", 1, 1),
                  _n("emo", "emotion_screen", 2, 1, fail_when="desperate > 2.5, hostile > 2.5"),
                  _n("ok", "output", 4, 1, name="result"),
                  _n("person", "approval", 3, 2, question="This reply reads as strongly emotional. Send it?"),
                  _n("held", "output", 4, 3, name="held")],
        "edges": [_e("q", "value", "reply", "prompt"), _e("reply", "text", "emo", "text"), _e("emo", "pass", "ok", "value"),
                  _e("emo", "fail", "person", "value"), _e("person", "approved", "ok", "value"),
                  _e("person", "rejected", "held", "value")]},
    "grounded-answer": {
        "name": "grounded-answer",
        "description": "Search your documents, answer from them, and pass the answer only if the sources support it.",
        "nodes": [_n("q", "input", 0, 1, name="question"), _n("find", "rag_search", 1, 1, collection="", k=4),
                  _n("answer", "llm", 2, 1, system="Answer from the context only, citing passages as [n]. Say so if the "
                                                   "context does not answer it."),
                  _n("check", "grounding", 3, 1, collection=""), _n("ok", "output", 4, 1, name="result"),
                  _n("unsupported", "output", 4, 2, name="unsupported")],
        "edges": [_e("q", "value", "find", "query"), _e("q", "value", "answer", "prompt"), _e("find", "text", "answer", "context"),
                  _e("answer", "text", "check", "answer"), _e("q", "value", "check", "question"),
                  _e("check", "pass", "ok", "value"), _e("check", "fail", "unsupported", "value")]},
}
