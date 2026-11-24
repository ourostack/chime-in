#!/usr/bin/env python3
"""Chime-in gate: a small decision model decides whether a proactive agent
should reply to a conversation, so the full LLM runs only when it should.

Uses Microsoft-Decision-1.1 on Microsoft Foundry. Standard library only.

  python3 chime_in.py demo                 open the demo page (start here)
  python3 chime_in.py run FILE             gate one conversation in the terminal
  python3 chime_in.py eval [--compare]     score the gate on labeled conversations
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
CONFIG_PATH = Path.home() / ".config" / "chime-in" / "config.json"

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


class ChimeError(RuntimeError):
    """Something the person running this can fix; the message says how."""


# ---------- Azure access ----------

_tokens: dict[str, tuple[str, float]] = {}


def _az_token(resource: str) -> str:
    cached = _tokens.get(resource)
    if cached and cached[1] > time.time() + 60:
        return cached[0]
    if not shutil.which("az"):
        raise ChimeError("Install the Azure CLI (https://aka.ms/azcli) and run `az login`.")
    out = subprocess.run(["az", "account", "get-access-token", "--resource", resource, "-o", "json"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise ChimeError(f"Couldn't get an Azure token; run `az login`.\n{out.stderr.strip()}")
    tok = json.loads(out.stdout)
    _tokens[resource] = (tok["accessToken"], float(tok.get("expires_on") or time.time() + 600))
    return tok["accessToken"]


def _http(method: str, url: str, headers: dict, body: dict | None = None) -> tuple[dict, float]:
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(8):
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json", **headers})
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
            detail = exc.read().decode(errors="replace")[:400]
            hint = ""
            if exc.code in (401, 403):
                hint = ("\nYour account can't call this resource. Ask its owner for the "
                        "'Cognitive Services User' role, or deploy the model in your own Foundry project.")
            raise ChimeError(f"HTTP {exc.code} from {url.split('?')[0]}: {detail}{hint}") from exc
        except urllib.error.URLError as exc:
            raise ChimeError(f"Couldn't reach {url.split('?')[0]}: {exc.reason}") from exc
    raise ChimeError("Still rate limited after 8 tries; raise the deployment's capacity in Foundry.")


# ---------- configuration ----------

def discover() -> dict:
    """Find a Microsoft-Decision deployment (and a chat model beside it) in any
    Foundry resource the signed-in account can see."""
    print("Looking for a Microsoft-Decision deployment in your Azure subscriptions...", file=sys.stderr)
    arm = {"Authorization": f"Bearer {_az_token('https://management.azure.com')}"}
    graph, _ = _http("POST", "https://management.azure.com/providers/Microsoft.ResourceGraph/resources"
                     "?api-version=2022-10-01", arm, {
                         "query": "resources | where type =~ 'microsoft.cognitiveservices/accounts' "
                                  "| where kind in~ ('AIServices','OpenAI') "
                                  "| project id, endpoint=tostring(properties.endpoint)",
                         "options": {"$top": 1000}})
    accounts = graph.get("data", [])

    def deployments(acct: dict) -> list[dict]:
        try:
            res, _ = _http("GET", f"https://management.azure.com{acct['id']}/deployments"
                           "?api-version=2024-10-01", arm)
        except ChimeError:
            return []
        return [{"endpoint": acct["endpoint"], "name": d["name"],
                 "model": d["properties"]["model"]["name"]} for d in res.get("value", [])]

    with ThreadPoolExecutor(16) as pool:
        found = [d for ds in pool.map(deployments, accounts) for d in ds]
    decisions = sorted((d for d in found if d["model"].lower().startswith("microsoft-decision")),
                       key=lambda d: d["model"], reverse=True)
    if not decisions:
        raise ChimeError(
            f"Checked {len(accounts)} Foundry resources and found no Microsoft-Decision deployment.\n"
            "Deploy one: https://ai.azure.com/catalog/models/Microsoft-Decision-1.1 > Deploy, "
            "then run this again.")
    pick = decisions[0]
    chats = [d for d in found if d["endpoint"] == pick["endpoint"]
             and d["model"].lower().startswith(("gpt", "o1", "o3", "o4", "phi", "mai"))
             and "image" not in d["model"].lower() and "embedding" not in d["model"].lower()]
    config = {"endpoint": pick["endpoint"].rstrip("/"), "decision_deployment": pick["name"],
              "chat_deployment": chats[0]["name"] if chats else ""}
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    print(f"Using {config['decision_deployment']} at {config['endpoint']}"
          + (f", chat model {config['chat_deployment']}" if config["chat_deployment"] else ", no chat model")
          + f".\nSaved to {CONFIG_PATH}; delete it to search again.", file=sys.stderr)
    return config


_config: dict | None = None


def config() -> dict:
    global _config
    if _config is None:
        saved = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
        if not os.environ.get("FOUNDRY_ENDPOINT") and not saved:
            saved = discover()
        _config = {
            "endpoint": (os.environ.get("FOUNDRY_ENDPOINT") or saved.get("endpoint", "")).rstrip("/"),
            "decision_deployment": os.environ.get("DECISION_DEPLOYMENT") or saved.get("decision_deployment") or "decision-1-1",
            "chat_deployment": os.environ.get("CHAT_DEPLOYMENT", saved.get("chat_deployment", "")),
        }
    return _config


def _foundry(path: str, body: dict) -> tuple[dict, float]:
    key = os.environ.get("FOUNDRY_API_KEY")
    auth = {"api-key": key} if key else \
        {"Authorization": f"Bearer {_az_token('https://cognitiveservices.azure.com')}"}
    return _http("POST", config()["endpoint"] + path, auth, body)


# ---------- the gate and the reply ----------

def gate_request(agent: str, conversation: list[dict], context: str = "") -> dict:
    return {
        "model": config()["decision_deployment"],
        "state": {"agent": agent, "agent_knows": context or "nothing beyond the conversation",
                  "conversation": conversation},
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
    }


def gate(agent: str, conversation: list[dict], context: str = "") -> dict:
    """One Decision-1.1 call: how should the agent contribute, and how much is it worth?"""
    request = gate_request(agent, conversation, context)
    payload, ms = _foundry("/providers/microsoft/v1/systemone", request)
    move, value = payload["answers"]["move"], payload["answers"]["value"]
    p_silent = move["probabilities"].get("stay_silent", 0.0)
    return {
        "move": move["choice"],
        "move_probabilities": move["probabilities"],
        "p_silent": p_silent,
        "value": value["score"],
        "value_probabilities": value.get("probabilities", {}),
        "reply": p_silent < MAX_SILENT_PROBABILITY and value["score"] >= MIN_VALUE_SCORE,
        "tokens": payload["usage"]["input_tokens"] + payload["usage"]["output_tokens"],
        "ms": ms,
        "request": request,
        "response": payload,
    }


def _transcript(conversation: list[dict]) -> str:
    return "\n".join(f"{m['from']}: {m['text']}" for m in conversation)


def respond(agent: str, conversation: list[dict], context: str = "", always_on: bool = False) -> dict:
    """The full LLM turn. With always_on, the LLM also decides whether to stay silent,
    which is how a proactive agent works without a gate."""
    deployment = config()["chat_deployment"]
    if not deployment:
        raise ChimeError("No chat model found beside the decision model; set CHAT_DEPLOYMENT to draft replies.")
    system = (f"You are {agent}\nYou know: {context or 'nothing beyond the conversation'}\n"
              "Reply to the conversation briefly, as a helpful teammate.")
    if always_on:
        system += (" If you have nothing worth adding, reply with exactly SILENT. "
                   "Stay silent for social chat, questions already answered, or questions for someone else.")
    payload, ms = _foundry(
        f"/openai/deployments/{deployment}/chat/completions?api-version=2024-10-21",
        {"messages": [{"role": "system", "content": system},
                      {"role": "user", "content": _transcript(conversation)}]},
    )
    text = payload["choices"][0]["message"]["content"].strip()
    return {"text": text, "silent": text.upper().startswith("SILENT"),
            "tokens": payload["usage"]["total_tokens"], "ms": ms, "model": deployment}


# ---------- commands ----------

def run_one(path: str) -> None:
    case = json.loads(Path(path).read_text())
    g = gate(case["agent"], case["conversation"], case.get("context", ""))
    print(f"move={g['move']} p_silent={g['p_silent']:.2f} value={g['value']:.2f}/3 "
          f"-> {'REPLY' if g['reply'] else 'stay silent'} ({g['ms']:.0f} ms, {g['tokens']} tokens)")
    if g["reply"] and config()["chat_deployment"]:
        r = respond(case["agent"], case["conversation"], case.get("context", ""))
        print(f"\n{r['text']}\n\n({r['ms']:.0f} ms, {r['tokens']} tokens)")


def evaluate(path: str, compare: bool) -> None:
    data = json.loads(Path(path).read_text())
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


def chat_repl() -> None:
    """Type a conversation one message at a time; after each, see whether the agent would chime in."""
    agent = "Atlas, the team's assistant. It can search the team's notes, customer records and calendars."
    print(f"The agent: {agent}")
    agent = input("Describe a different agent, or press Enter to keep this one: ").strip() or agent
    context = input("What does the agent know? (notes, calendar, records; Enter for nothing): ").strip()
    print("\nNow type the chat, one message per line, as  Name: message")
    print("After each message, Decision-1.1 says whether the agent would chime in.")
    print("/reset starts a new chat, /why shows the last full answer, Ctrl+D quits.\n")
    conversation: list[dict] = []
    last: dict | None = None
    short = agent.split(",")[0].split(".")[0]
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line == "/reset":
            conversation.clear()
            print("  (new chat)\n")
            continue
        if line == "/why":
            print(json.dumps(last["response"] if last else {}, indent=2) + "\n")
            continue
        name, _, text = line.partition(":")
        conversation.append({"from": name.strip(), "text": text.strip()} if text else {"from": "Someone", "text": line})
        try:
            last = gate(agent, conversation, context)
        except ChimeError as exc:
            print(f"  error: {exc}\n")
            conversation.pop()
            continue
        top = sorted(last["move_probabilities"].items(), key=lambda kv: -kv[1])[:2]
        moves = ", ".join(f"{k.replace('_', ' ')} {p:.0%}" for k, p in top)
        why = f"{moves}; value {last['value']:.1f}/3; {last['ms']:.0f} ms"
        if not last["reply"]:
            print(f"  [{short} stays silent: {why}]\n")
            continue
        print(f"  [{short} chimes in: {why}]")
        if config()["chat_deployment"]:
            try:
                r = respond(agent, conversation, context)
            except ChimeError as exc:
                print(f"  error: {exc}\n")
                continue
            print(f"{short}: {r['text']}\n")
            conversation.append({"from": short, "text": r["text"]})
        else:
            print()


class _DemoHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        pass

    def _send(self, status: int, body: bytes, kind: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj: dict) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:
        if self.path == "/":
            self._send(200, (HERE / "demo.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/setup":
            data = json.loads((HERE / "examples" / "eval.json").read_text())
            self._json(200, {**data, "config": config(), "moves": MOVES, "value_levels": VALUE_LEVELS,
                             "max_silent": MAX_SILENT_PROBABILITY, "min_value": MIN_VALUE_SCORE})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        try:
            if self.path == "/gate":
                self._json(200, gate(body["agent"], body["conversation"], body.get("context", "")))
            elif self.path == "/reply":
                self._json(200, respond(body["agent"], body["conversation"], body.get("context", ""),
                                        always_on=body.get("always_on", False)))
            else:
                self._send(404, b"not found", "text/plain")
        except ChimeError as exc:
            self._json(502, {"error": str(exc)})


def demo(port: int, open_browser: bool) -> None:
    config()
    server = ThreadingHTTPServer(("127.0.0.1", port), _DemoHandler)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Demo running at {url} (Ctrl+C to stop)", file=sys.stderr)
    if open_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="open the demo page (start here)")
    d.add_argument("--port", type=int, default=8765)
    d.add_argument("--no-browser", action="store_true")
    sub.add_parser("chat", help="type a chat in the terminal and watch the agent decide after each message")
    r = sub.add_parser("run", help="gate one conversation, and draft a reply if it passes")
    r.add_argument("file", help="JSON with agent, conversation and optional context")
    e = sub.add_parser("eval", help="score the gate against labeled conversations")
    e.add_argument("file", nargs="?", default=str(HERE / "examples" / "eval.json"))
    e.add_argument("--compare", action="store_true",
                   help="also run an always-on LLM on every conversation and compare tokens")
    args = p.parse_args()
    try:
        if args.cmd == "demo":
            demo(args.port, not args.no_browser)
        elif args.cmd == "chat":
            chat_repl()
        elif args.cmd == "run":
            run_one(args.file)
        else:
            evaluate(args.file, args.compare)
    except ChimeError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
