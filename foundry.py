"""Plumbing: how to call models on Microsoft Foundry. Read gate.py for the idea.

Finds your deployments through `az login`, signs each request, and retries
when Foundry says to slow down.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path.home() / ".config" / "chime-in" / "config.json"


class FoundryError(RuntimeError):
    """Something the person running this can fix; the message says how."""


@dataclass
class Call:
    """One request to a model: what went out, what came back, how long it took, what it cost."""
    request: dict
    response: dict
    ms: float
    tokens: int
    deployment: str


# ---------- the two kinds of model call ----------

def choose_one_of(options: dict[str, str], instructions: str) -> dict:
    """A decision-model question with a fixed set of answers."""
    return {"type": "choice", "instructions": instructions, "criteria": options}


def score_on_a_scale(levels: list[str], instructions: str) -> dict:
    """A decision-model question answered with a score from 0 to len(levels) - 1."""
    return {"type": "score", "instructions": instructions, "criteria": levels}


def ask_decision_model(situation: dict, questions: dict) -> tuple[dict, Call]:
    """Ask Microsoft-Decision-1.1 questions about a situation; returns its answers by question name."""
    deployment = settings()["decision_deployment"]
    request = {"model": deployment, "state": situation, "questions": questions}
    response, ms = _post("/providers/microsoft/v1/systemone", request)
    usage = response.get("usage", {})
    tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    return response["answers"], Call(request, response, ms, tokens, deployment)


def ask_chat_model(instructions: str, conversation: list[dict]) -> tuple[str, Call]:
    """Ask the chat model to respond to a conversation; returns its text."""
    deployment = settings()["chat_deployment"]
    if not deployment:
        raise FoundryError("No chat model found beside the decision model; "
                           "deploy one in the same Foundry project, or set CHAT_DEPLOYMENT.")
    transcript = "\n".join(f"{m['from']}: {m['text']}" for m in conversation)
    request = {"messages": [{"role": "system", "content": instructions},
                            {"role": "user", "content": transcript}]}
    response, ms = _post(f"/openai/deployments/{deployment}/chat/completions?api-version=2024-10-21", request)
    text = response["choices"][0]["message"]["content"].strip()
    return text, Call(request, response, ms, response["usage"]["total_tokens"], deployment)


# ---------- which deployments to use ----------

_settings: dict | None = None


def settings() -> dict:
    """Environment variables win; otherwise the saved choice; otherwise search Azure once."""
    global _settings
    if _settings is None:
        saved = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
        if not os.environ.get("FOUNDRY_ENDPOINT") and not saved:
            saved = find_deployments()
        _settings = {
            "endpoint": (os.environ.get("FOUNDRY_ENDPOINT") or saved.get("endpoint", "")).rstrip("/"),
            "decision_deployment": os.environ.get("DECISION_DEPLOYMENT") or saved.get("decision_deployment") or "decision-1-1",
            "chat_deployment": os.environ.get("CHAT_DEPLOYMENT", saved.get("chat_deployment", "")),
        }
    return _settings


def find_deployments() -> dict:
    """Look through every Foundry resource you can see for a Microsoft-Decision deployment,
    and a chat model beside it. Saves what it finds so it only searches once."""
    print("Looking for a Microsoft-Decision deployment in your Azure subscriptions...", file=sys.stderr)
    azure = {"Authorization": f"Bearer {_token('https://management.azure.com')}"}
    found, _ = _http("POST", "https://management.azure.com/providers/Microsoft.ResourceGraph/resources"
                     "?api-version=2022-10-01", azure, {
                         "query": "resources | where type =~ 'microsoft.cognitiveservices/accounts' "
                                  "| where kind in~ ('AIServices','OpenAI') "
                                  "| project id, endpoint=tostring(properties.endpoint)",
                         "options": {"$top": 1000}})
    resources = found.get("data", [])

    def deployments_in(resource: dict) -> list[dict]:
        try:
            listed, _ = _http("GET", f"https://management.azure.com{resource['id']}/deployments"
                              "?api-version=2024-10-01", azure)
        except FoundryError:
            return []
        return [{"endpoint": resource["endpoint"], "name": d["name"], "model": d["properties"]["model"]["name"]}
                for d in listed.get("value", [])]

    with ThreadPoolExecutor(16) as pool:
        deployments = [d for ds in pool.map(deployments_in, resources) for d in ds]

    decision = max((d for d in deployments if d["model"].lower().startswith("microsoft-decision")),
                   key=lambda d: d["model"], default=None)
    if not decision:
        raise FoundryError(
            f"Checked {len(resources)} Foundry resources and found no Microsoft-Decision deployment.\n"
            "Deploy one: https://ai.azure.com/catalog/models/Microsoft-Decision-1.1 > Deploy, "
            "then run this again.")
    chat = next((d for d in deployments if d["endpoint"] == decision["endpoint"]
                 and d["model"].lower().startswith(("gpt", "o1", "o3", "o4", "phi", "mai"))
                 and not any(w in d["model"].lower() for w in ("image", "embedding", "audio", "realtime"))), None)

    chosen = {"endpoint": decision["endpoint"].rstrip("/"), "decision_deployment": decision["name"],
              "chat_deployment": chat["name"] if chat else ""}
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(chosen, indent=2))
    print(f"Using {chosen['decision_deployment']} at {chosen['endpoint']}"
          + (f", chat model {chosen['chat_deployment']}" if chat else ", no chat model")
          + f".\nSaved to {CONFIG_PATH}; delete it to search again.", file=sys.stderr)
    return chosen


# ---------- signing in and sending requests ----------

_tokens: dict[str, tuple[str, float]] = {}


def _token(resource: str) -> str:
    cached = _tokens.get(resource)
    if cached and cached[1] > time.time() + 60:
        return cached[0]
    if not shutil.which("az"):
        raise FoundryError("Install the Azure CLI (https://aka.ms/azcli) and run `az login`.")
    out = subprocess.run(["az", "account", "get-access-token", "--resource", resource, "-o", "json"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise FoundryError(f"Couldn't get an Azure token; run `az login`.\n{out.stderr.strip()}")
    token = json.loads(out.stdout)
    _tokens[resource] = (token["accessToken"], float(token.get("expires_on") or time.time() + 600))
    return token["accessToken"]


def _post(path: str, body: dict) -> tuple[dict, float]:
    key = os.environ.get("FOUNDRY_API_KEY")
    auth = {"api-key": key} if key else {"Authorization": f"Bearer {_token('https://cognitiveservices.azure.com')}"}
    return _http("POST", settings()["endpoint"] + path, auth, body)


def _http(method: str, url: str, headers: dict, body: dict | None = None) -> tuple[dict, float]:
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(8):
        request = urllib.request.Request(url, data=data, method=method,
                                         headers={"Content-Type": "application/json", **headers})
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read()), (time.perf_counter() - start) * 1000
        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt < 7:
                wait_ms = error.headers.get("retry-after-ms")
                wait = float(wait_ms) / 1000 if wait_ms else float(error.headers.get("Retry-After") or 2 ** attempt)
                print(f"  Foundry says slow down; retrying in {wait:.0f} s", file=sys.stderr)
                time.sleep(min(wait, 90))
                continue
            detail = error.read().decode(errors="replace")[:400]
            hint = ("\nYour account can't call this resource. Ask its owner for the "
                    "'Cognitive Services User' role, or deploy the model in your own Foundry project."
                    if error.code in (401, 403) else "")
            raise FoundryError(f"HTTP {error.code} from {url.split('?')[0]}: {detail}{hint}") from error
        except urllib.error.URLError as error:
            raise FoundryError(f"Couldn't reach {url.split('?')[0]}: {error.reason}") from error
    raise FoundryError("Still rate limited after 8 tries; raise the deployment's capacity in Foundry.")
