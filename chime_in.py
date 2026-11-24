#!/usr/bin/env python3
"""Try the chime-in gate. The idea itself is in gate.py.

  python3 chime_in.py demo               open the demo page (start here)
  python3 chime_in.py chat               type a chat; see the decision after each message
  python3 chime_in.py run FILE           decide on one conversation file
  python3 chime_in.py eval [--compare]   score the gate on labeled conversations
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import median

import gate
from foundry import FoundryError, settings

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE / "examples" / "eval.json"
DEFAULT_AGENT = json.loads(EXAMPLES.read_text())["agent"]


def summary(d: gate.Decision) -> str:
    kind = f"; would {d.kind.replace('_', ' ')}" if d.reply else ""
    return f"value {d.value:.1f}/3{kind}; {d.call.ms:.0f} ms"


# ---------- python3 chime_in.py chat ----------

def chat() -> None:
    print(f"The agent: {DEFAULT_AGENT}")
    agent = input("Describe a different agent, or press Enter to keep this one: ").strip() or DEFAULT_AGENT
    agent_knows = input("What does the agent know? (notes, calendar, records; Enter for nothing): ").strip()
    name = agent.split(",")[0].split(".")[0]
    print("\nNow type the chat, one message per line, as  Name: message")
    print("After each message, Decision-1.1 says whether the agent would chime in.")
    print("/reset starts a new chat, /why shows the last full answer, Ctrl+D quits.\n")

    conversation: list[dict] = []
    last: gate.Decision | None = None
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
            print(json.dumps(last.call.response if last else {}, indent=2) + "\n")
            continue

        sender, _, text = line.partition(":")
        conversation.append({"from": sender.strip(), "text": text.strip()} if text else {"from": "Someone", "text": line})
        try:
            last = gate.should_chime_in(agent, conversation, agent_knows)
            if not last.reply:
                print(f"  [{name} stays silent: {summary(last)}]\n")
                continue
            print(f"  [{name} chimes in: {summary(last)}]")
            reply, _ = gate.write_reply(agent, conversation, agent_knows, last.kind)
            print(f"{name}: {reply}\n")
            conversation.append({"from": name, "text": reply})
        except FoundryError as error:
            print(f"  error: {error}\n")


# ---------- python3 chime_in.py run FILE ----------

def run(path: str) -> None:
    case = json.loads(Path(path).read_text())
    d = gate.should_chime_in(case["agent"], case["conversation"], case.get("context", ""))
    print(f"{'CHIME IN' if d.reply else 'stay silent'}: {summary(d)}, {d.call.tokens} tokens")
    if d.reply and settings()["chat_deployment"]:
        reply, call = gate.write_reply(case["agent"], case["conversation"], case.get("context", ""), d.kind)
        print(f"\n{reply}\n\n({call.ms:.0f} ms, {call.tokens} tokens)")


# ---------- python3 chime_in.py eval ----------

def evaluate(path: str, compare: bool) -> None:
    data = json.loads(Path(path).read_text())
    agent, cases = data["agent"], data["cases"]
    gate_right, gate_ms, gate_tokens, gated_llm_tokens = 0, [], 0, 0
    llm_right, llm_ms, llm_tokens = 0, [], 0
    for case in cases:
        args = (agent, case["conversation"], case.get("context", ""))
        d = gate.should_chime_in(*args)
        gate_right += d.reply == case["should_reply"]
        gate_ms.append(d.call.ms)
        gate_tokens += d.call.tokens
        line = (f"{'ok  ' if d.reply == case['should_reply'] else 'MISS'} {case['name']:<32} "
                f"{'reply ' if d.reply else 'silent'} (want {'reply' if case['should_reply'] else 'silent'})  {summary(d)}")
        if compare:
            if d.reply:
                gated_llm_tokens += gate.write_reply(*args, d.kind)[1].tokens
            text, call = gate.let_the_llm_decide(*args)
            llm_right += (text is not None) == case["should_reply"]
            llm_ms.append(call.ms)
            llm_tokens += call.tokens
            line += f"  | LLM alone: {'reply' if text else 'silent'}"
        print(line)

    print(f"\nGate: right on {gate_right}/{len(cases)}, median {median(gate_ms):.0f} ms, "
          f"{gate_tokens} Decision-1.1 tokens in all.")
    if compare:
        print(f"LLM deciding alone: right on {llm_right}/{len(cases)}, median {median(llm_ms):.0f} ms.")
        print(f"LLM tokens: {llm_tokens} when it reads every chat, {gated_llm_tokens} when the gate decides first.")


# ---------- python3 chime_in.py demo ----------

class DemoPage(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/":
            self.send(200, (HERE / "demo.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/setup":
            self.send_json(200, {
                **json.loads(EXAMPLES.read_text()),
                "settings": settings(),
                "how_valuable": gate.HOW_VALUABLE,
                "value_at_least": gate.REPLY_IF_VALUE_AT_LEAST,
                "kind_of_reply": gate.KIND_OF_REPLY,
            })
        else:
            self.send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        args = (body["agent"], body["conversation"], body.get("context", ""))
        try:
            if self.path == "/decide":
                d = gate.should_chime_in(*args)
                self.send_json(200, {"reply": d.reply, "value": d.value, "kind": d.kind,
                                     "kind_probabilities": d.kind_probabilities, **call_json(d.call)})
            elif self.path == "/reply":
                text, call = gate.write_reply(*args, body.get("kind", ""))
                self.send_json(200, {"text": text, **call_json(call)})
            elif self.path == "/llm-alone":
                text, call = gate.let_the_llm_decide(*args)
                self.send_json(200, {"text": text, **call_json(call)})
            else:
                self.send(404, b"not found", "text/plain")
        except FoundryError as error:
            self.send_json(502, {"error": str(error)})

    def send(self, status: int, body: bytes, kind: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, obj: dict) -> None:
        self.send(status, json.dumps(obj).encode(), "application/json")

    def log_message(self, *_args) -> None:
        pass


def call_json(call) -> dict:
    return {"ms": call.ms, "tokens": call.tokens, "deployment": call.deployment,
            "request": call.request, "response": call.response}


def demo(port: int, open_browser: bool) -> None:
    settings()
    server = ThreadingHTTPServer(("127.0.0.1", port), DemoPage)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Demo running at {url} (Ctrl+C to stop)", file=sys.stderr)
    if open_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    d = commands.add_parser("demo", help="open the demo page (start here)")
    d.add_argument("--port", type=int, default=8765)
    d.add_argument("--no-browser", action="store_true")
    commands.add_parser("chat", help="type a chat; see the decision after each message")
    r = commands.add_parser("run", help="decide on one conversation file")
    r.add_argument("file")
    e = commands.add_parser("eval", help="score the gate on labeled conversations")
    e.add_argument("file", nargs="?", default=str(EXAMPLES))
    e.add_argument("--compare", action="store_true", help="also let the LLM decide alone on every chat")
    args = parser.parse_args()
    try:
        if args.command == "demo":
            demo(args.port, not args.no_browser)
        elif args.command == "chat":
            chat()
        elif args.command == "run":
            run(args.file)
        else:
            evaluate(args.file, args.compare)
    except FoundryError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
