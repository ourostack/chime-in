# chime-in

A proof of concept: a small decision model decides whether a proactive agent should reply to a conversation, so the expensive LLM runs only when a reply is worth sending.

A proactive agent usually reads every message with a full LLM turn just to decide whether to speak, so it pays for every message even when it stays silent. Here, one call to [Microsoft-Decision-1.1](https://learn.microsoft.com/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision) asks two questions about the conversation:

- **What should the agent do now?** A choice of `stay_silent`, `answer_question`, `share_context`, `correct_mistake` or `do_task`, with a probability for each.
- **How valuable would a reply be?** A score from 0 (noise) to 3 (people need it now).

The LLM drafts a reply only when `stay_silent` is under 50% and the value is at least 1.5.

**Read the code:** [`gate.py`](gate.py) is the whole idea in about 100 lines: the two questions, the rule and the reply. [`foundry.py`](foundry.py) is the plumbing for calling Foundry, and [`chime_in.py`](chime_in.py) holds the commands below.

## Try it

You need Python 3.10 or later and the [Azure CLI](https://aka.ms/azcli). Nothing to install.

1. Deploy the model once: open [Microsoft-Decision-1.1 in the Foundry catalog](https://ai.azure.com/catalog/models/Microsoft-Decision-1.1), choose **Deploy** and pick any project. It's generally available, so you don't need access to any internal resource. Set the capacity above 1, because 1 allows only one request a minute. To see draft replies too, deploy a chat model (such as a GPT model) in the same project.
2. Run:

```sh
az login
git clone https://github.com/ourostack/chime-in && cd chime-in
python3 chime_in.py demo
```

The first run finds your deployment by itself and remembers it in `~/.config/chime-in/config.json`. A page opens in your browser: pick an example chat or type your own, press **Ask Decision-1.1**, and it shows the verdict, the probability behind every option, the reply the chat model wrote, and the exact request and response.

To watch it decide message by message instead, chat in the terminal:

```sh
python3 chime_in.py chat
```

```text
> Maya: lunch anyone?
  [Atlas stays silent: stay silent 98%, share context 1%; value 0.3/3; 448 ms]
> Leo: yes, tacos at 12:30
  [Atlas stays silent: stay silent 96%, share context 4%; value 0.2/3; 334 ms]
> Leo: oh and see you all at the Contoso review on Wednesday
  [Atlas chimes in: correct mistake 88%, share context 8%; value 2.4/3; 368 ms]
Atlas: Small update: the Contoso review was moved to Thursday at 10:00 PT.
```

(Here the agent was told the review moved to Thursday.)

## Other commands

- `python3 chime_in.py run examples/one.json` gates one conversation file and prints the result.
- `python3 chime_in.py eval --compare` scores the gate on the labeled chats in [`examples/eval.json`](examples/eval.json) and, with `--compare`, runs an LLM on every chat to compare.

A conversation file has the agent's description, what the agent knows (`context`) and the messages. The examples are made up; use chats shaped like yours before trusting the thresholds.

To skip the search, or to use an API key, set `FOUNDRY_ENDPOINT`, `DECISION_DEPLOYMENT`, `CHAT_DEPLOYMENT` and `FOUNDRY_API_KEY`.

## First results

On the 12 made-up conversations in [`examples/eval.json`](examples/eval.json), on 2026-11-24 with Decision-1.1 and gpt-5.2-chat in East US 2:

| | Right calls | Median time per message |
|---|---|---|
| Decision-1.1 gate | 12 of 12 | 409 ms |
| Always-on LLM deciding for itself | 12 of 12 | 1,206 ms |

Twelve easy cases show the gate works, not that it beats an LLM; harder, real-shaped conversations are the next test. The token counts in this toy don't settle cost either: the gate's prompt carries the option descriptions every time, while the always-on LLM here has a one-line prompt. A real proactive agent carries its full instructions, tools and memory on every turn, which is the cost the gate avoids.
