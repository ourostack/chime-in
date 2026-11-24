#!/usr/bin/env python3
"""Chime-in gate: a small decision model decides whether a proactive agent
should reply to a conversation, so the full LLM runs only when it should.

Uses Microsoft-Decision-1.1 on Microsoft Foundry. Standard library only.
Run `python3 chime_in.py --help` for usage.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from statistics import median

MOVES = {
    "stay_silent": "Nothing to add: the conversation is social, already answered, "
    "addressed to a specific person, or the agent knows nothing useful about it",
    "answer_question": "Someone asked an open question the agent can answer from what it knows",
    "share_context": "The agent knows something relevant the people talking don't seem to have",
    "correct_mistake": "Someone stated something the agent knows is wrong",
    "do_task": "Someone asked the agent itself to do something",
}

VALUE_LEVELS = [
    "None: a reply would be noise",
    "Low: mildly helpful, people would not miss it",
    "Medium: saves someone real time or effort",
    "High: people need this now and would be worse off without it",
]

# A reply runs only when both say so.
MAX_SILENT_PROBABILITY = 0.5
MIN_VALUE_SCORE = 1.5  # on the 0-3 scale above

_token_cache: dict[str, float | str] = {}


def _auth_header() -> dict[str, str]:
    key = os.environ.get("FOUNDRY_API_KEY")
    if key:
        return {"api-key": key}
    if _token_cache.get("expires", 0) > time.time() + 60:  # type: ignore[operator]
        return {"Authorization": f"Bearer {_token_cache['token']}"}
    if not shutil.which("az"):
        sys.exit("Set FOUNDRY_API_KEY, or install the Azure CLI and run `az login`.")
    out = subprocess.run(
        ["az", "account", "get-access-token", "--resource",
         "https://cognitiveservices.azure.com", "-o", "json"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        sys.exit(f"Couldn't get an Entra token; run `az login`.\n{out.stderr.strip()}")
    tok = json.loads(out.stdout)
    _token_cache["token"] = tok["accessToken"]
    _token_cache["expires"] = float(tok.get("expires_on") or time.time() + 600)
    return {"Authorization": f"Bearer {tok['accessToken']}"}


def _post(path: str, body: dict) -> tuple[dict, float]:
    endpoint = os.environ.get("FOUNDRY_ENDPOINT", "").rstrip("/")
    if not endpoint:
        sys.exit("Set FOUNDRY_ENDPOINT to your Foundry resource, e.g. "
                 "https://<name>.cognitiveservices.azure.com")
    req = urllib.request.Request(
        endpoint + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **_auth_header()},
        method="POST",
    )
    for attempt in range(8):
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read()), (time.perf_counter() - start) * 1000
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 7:
                wait_ms = exc.headers.get("retry-after-ms")
                wait = float(wait_ms) / 1000 if wait_ms else float(exc.headers.get("Retry-After") or 2 ** attempt)
                print(f"  rate limited; retrying in {wait:.0f} s", file=sys.stderr)
                time.sleep(min(wait, 90))
                continue
            sys.exit(f"{path} returned HTTP {exc.code}: {exc.read().decode(errors='replace')[:500]}")
    raise AssertionError("unreachable")


def gate(agent: str, conversation: list[dict], context: str = "") -> dict:
    """One Decision-1.1 call: how should the agent contribute, and how much is it worth?"""
    state = {"agent": agent, "agent_knows": context or "nothing beyond the conversation",
             "conversation": conversation}
    payload, ms = _post("/providers/microsoft/v1/systemone", {
        "model": os.environ.get("DECISION_DEPLOYMENT", "decision-1-1"),
        "state": state,
        "questions": {
            "move": {
                "type": "choice",
                "instructions": "The agent is a quiet member of this conversation. Based on the "
                "latest messages, what should it do now?",
                "criteria": MOVES,
            },
            "value": {
                "type": "score",
                "instructions": "If the agent replied now, how valuable would that reply be to "
                "the people in the conversation?",
                "criteria": VALUE_LEVELS,
            },
        },
    })
    move, value = payload["answers"]["move"], payload["answers"]["value"]
    p_silent = move["probabilities"].get("stay_silent", 0.0)
    return {
        "move": move["choice"],
        "p_silent": p_silent,
        "value": value["score"],
        "reply": p_silent < MAX_SILENT_PROBABILITY and value["score"] >= MIN_VALUE_SCORE,
        "tokens": payload["usage"]["input_tokens"] + payload["usage"]["output_tokens"],
        "ms": ms,
    }


def _transcript(conversation: list[dict]) -> str:
    return "\n".join(f"{m['from']}: {m['text']}" for m in conversation)


def respond(agent: str, conversation: list[dict], context: str = "", always_on: bool = False) -> dict:
    """The full LLM turn. With always_on, the LLM also decides whether to stay silent,
    which is how a proactive agent works without a gate."""
    deployment = os.environ.get("CHAT_DEPLOYMENT")
    if not deployment:
        sys.exit("Set CHAT_DEPLOYMENT to a chat model deployment to draft replies.")
    system = (f"You are {agent}\nYou know: {context or 'nothing beyond the conversation'}\n"
              "Reply to the conversation briefly, as a helpful teammate.")
    if always_on:
        system += (" If you have nothing worth adding, reply with exactly SILENT. "
                   "Stay silent for social chat, questions already answered, or questions for someone else.")
    payload, ms = _post(
        f"/openai/deployments/{deployment}/chat/completions?api-version=2024-10-21",
        {"messages": [{"role": "system", "content": system},
                      {"role": "user", "content": _transcript(conversation)}]},
    )
    text = payload["choices"][0]["message"]["content"].strip()
    return {"text": text, "silent": text.upper().startswith("SILENT"),
            "tokens": payload["usage"]["total_tokens"], "ms": ms}


def run_one(path: str) -> None:
    with open(path) as f:
        case = json.load(f)
    g = gate(case["agent"], case["conversation"], case.get("context", ""))
    print(f"move={g['move']} p_silent={g['p_silent']:.2f} value={g['value']:.2f}/3 "
          f"-> {'REPLY' if g['reply'] else 'stay silent'} ({g['ms']:.0f} ms, {g['tokens']} tokens)")
    if g["reply"] and os.environ.get("CHAT_DEPLOYMENT"):
        r = respond(case["agent"], case["conversation"], case.get("context", ""))
        print(f"\n{r['text']}\n\n({r['ms']:.0f} ms, {r['tokens']} tokens)")


def evaluate(path: str, compare: bool) -> None:
    with open(path) as f:
        data = json.load(f)
    agent, cases = data["agent"], data["cases"]
    rows, gate_ms, gate_tokens, gated_llm_tokens = [], [], 0, 0
    base_tokens, base_ms, base_right = 0, [], 0
    for case in cases:
        g = gate(agent, case["conversation"], case.get("context", ""))
        gate_ms.append(g["ms"])
        gate_tokens += g["tokens"]
        right = g["reply"] == case["should_reply"]
        line = (f"{'ok ' if right else 'MISS'} {case['name']:<34} {g['move']:<16} "
                f"silent={g['p_silent']:.2f} value={g['value']:.2f} -> "
                f"{'reply' if g['reply'] else 'silent'} (want {'reply' if case['should_reply'] else 'silent'})")
        if compare:
            if g["reply"]:
                gated_llm_tokens += respond(agent, case["conversation"], case.get("context", ""))["tokens"]
            b = respond(agent, case["conversation"], case.get("context", ""), always_on=True)
            base_tokens += b["tokens"]
            base_ms.append(b["ms"])
            base_right += (not b["silent"]) == case["should_reply"]
            line += f" | always-on LLM: {'silent' if b['silent'] else 'reply'}"
        rows.append(right)
        print(line)
    n = len(cases)
    print(f"\nGate agreed with the label on {sum(rows)}/{n}; "
          f"median {median(gate_ms):.0f} ms, {gate_tokens} decision tokens in all.")
    if compare:
        print(f"Always-on LLM agreed on {base_right}/{n}; median {median(base_ms):.0f} ms per message.")
        print(f"LLM tokens: {base_tokens} always-on vs {gated_llm_tokens} gated, "
              f"plus {gate_tokens} Decision-1.1 tokens. Compare cost with your own prices "
              "and your agent's real prompt size.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="gate one conversation, and draft a reply if it passes")
    r.add_argument("file", help="JSON with agent, conversation and optional context")
    e = sub.add_parser("eval", help="score the gate against labeled conversations")
    e.add_argument("file", nargs="?", default=os.path.join(os.path.dirname(__file__), "examples", "eval.json"))
    e.add_argument("--compare", action="store_true",
                   help="also run an always-on LLM on every conversation and compare tokens")
    args = p.parse_args()
    if args.cmd == "run":
        run_one(args.file)
    else:
        evaluate(args.file, args.compare)


if __name__ == "__main__":
    main()
