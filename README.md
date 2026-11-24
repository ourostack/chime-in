# chime-in

Labs, unsupported. A proof of concept: a small decision model decides whether a proactive agent should reply to a conversation, so the expensive LLM runs only when a reply is worth sending.

A proactive agent usually reads every message with a full LLM turn just to decide whether to speak, so it pays for every message even when it stays silent. Here, one call to [Microsoft-Decision-1.1](https://learn.microsoft.com/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision) asks two questions about the conversation:

- **How should the agent contribute?** A choice of `stay_silent`, `answer_question`, `share_context`, `correct_mistake` or `do_task`, with a probability for each.
- **How valuable would a reply be?** A score from 0 (noise) to 3 (people need it now).

The LLM drafts a reply only when `stay_silent` is under 50% and the value is at least 1.5. Both thresholds are at the top of [`chime_in.py`](chime_in.py).

## Set up

1. In [Foundry](https://ai.azure.com), open a project, find **Microsoft-Decision-1.1** in the model catalog and deploy it. Decision-1.1 is generally available, so you don't need access to any internal resource. Give the deployment enough rate limit for your test: a capacity of 1 allows one request a minute.
2. To draft replies too, deploy a chat model on the same resource.
3. Sign in with `az login`, or set `FOUNDRY_API_KEY` if the resource allows keys.
4. Set where to call:

```sh
export FOUNDRY_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com
export DECISION_DEPLOYMENT=<your Decision-1.1 deployment name>   # default: decision-1-1
export CHAT_DEPLOYMENT=<your chat model deployment name>          # optional
```

Python 3.10 or later; no packages to install.

## Use it

Gate one conversation, and draft a reply if it passes:

```sh
python3 chime_in.py run examples/one.json
```

```text
move=correct_mistake p_silent=0.01 value=2.64/3 -> REPLY (335 ms, 335 tokens)

Small update: the Contoso review was moved to Thursday at 10:00 PT. Contoso confirmed by email.
```

Score the gate against labeled conversations, and with `--compare` also run an always-on LLM on each one to compare tokens:

```sh
python3 chime_in.py eval --compare
```

A conversation file has the agent's description, what the agent knows (`context`) and the messages. See [`examples/`](examples). The examples are made up; replace them with conversations shaped like yours before trusting the thresholds.

## First results

On the 12 made-up conversations in [`examples/eval.json`](examples/eval.json), on 2026-11-24 with Decision-1.1 and gpt-5.2-chat in East US 2:

| | Right calls | Median time per message |
|---|---|---|
| Decision-1.1 gate | 12 of 12 | 409 ms |
| Always-on LLM deciding for itself | 12 of 12 | 1,206 ms |

Twelve easy cases show the gate works, not that it beats an LLM; harder, real-shaped conversations are the next test. The token counts in this toy don't settle cost either: the gate's prompt carries the option descriptions every time, while the always-on LLM here has a one-line prompt. A real proactive agent carries its full instructions, tools and memory on every turn, which is the cost the gate avoids.
