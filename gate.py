"""The whole idea, in one file.

A proactive agent sees every message in a conversation. Today it runs a full
LLM turn on each one just to decide whether to speak, and pays for that turn
even when it stays silent.

Instead, ask a small decision model (Microsoft-Decision-1.1) two questions
first, and run the LLM only when both answers say "reply".
"""

from __future__ import annotations

from dataclasses import dataclass

from foundry import Call, ask_chat_model, ask_decision_model, choose_one_of, score_on_a_scale

# Question 1: what should the agent do now?
# The decision model picks one option and gives a probability for every option.
WHAT_TO_DO = {
    "stay_silent": "Nothing to add: the conversation is social, already answered, "
    "addressed to a specific person, or the agent knows nothing useful about it",
    "answer_question": "Someone asked an open question the agent can answer from what it knows",
    "share_context": "The agent knows something relevant the people talking don't seem to have",
    "correct_mistake": "Someone stated something the agent knows is wrong",
    "do_task": "Someone asked the agent itself to do something",
}

# Question 2: how valuable would a reply be?
# The decision model gives a score from 0 (the first level) to 3 (the last).
HOW_VALUABLE = [
    "None: a reply would be noise",
    "Low: mildly helpful, people would not miss it",
    "Medium: saves someone real time or effort",
    "High: people need this now and would be worse off without it",
]

# The rule: reply only if staying silent is unlikely AND the reply is worth sending.
STAY_SILENT_BELOW = 0.5
VALUE_AT_LEAST = 1.5


@dataclass
class Decision:
    reply: bool                       # the verdict: should the agent chime in?
    what_to_do: str                   # the option the model picked for question 1
    probabilities: dict[str, float]   # question 1: the probability of every option
    chance_of_silence: float          # question 1: the probability of "stay_silent"
    value: float                      # question 2: the score, 0 to 3
    call: Call                        # the exact request, response, time and tokens


def should_chime_in(agent: str, conversation: list[dict], agent_knows: str = "") -> Decision:
    """One call to the decision model answers both questions."""
    answers, call = ask_decision_model(
        situation={
            "agent": agent,
            "agent_knows": agent_knows or "nothing beyond the conversation",
            "conversation": conversation,
        },
        questions={
            "what_to_do": choose_one_of(
                WHAT_TO_DO,
                "The agent is a quiet member of this conversation. "
                "Based on the latest messages, what should it do now?"),
            "how_valuable": score_on_a_scale(
                HOW_VALUABLE,
                "If the agent replied now, how valuable would that reply be "
                "to the people in the conversation?"),
        },
    )

    what_to_do = answers["what_to_do"]
    chance_of_silence = what_to_do["probabilities"].get("stay_silent", 0.0)
    value = answers["how_valuable"]["score"]

    return Decision(
        reply=chance_of_silence < STAY_SILENT_BELOW and value >= VALUE_AT_LEAST,
        what_to_do=what_to_do["choice"],
        probabilities=what_to_do["probabilities"],
        chance_of_silence=chance_of_silence,
        value=value,
        call=call,
    )


def write_reply(agent: str, conversation: list[dict], agent_knows: str = "") -> tuple[str, Call]:
    """Only after the gate says yes: the full LLM writes the reply."""
    return ask_chat_model(
        instructions=f"You are {agent}\n"
                     f"You know: {agent_knows or 'nothing beyond the conversation'}\n"
                     "Reply to the conversation briefly, as a helpful teammate.",
        conversation=conversation,
    )


def let_the_llm_decide(agent: str, conversation: list[dict], agent_knows: str = "") -> tuple[str | None, Call]:
    """For comparison, what proactive agents do today: the full LLM reads every
    message and decides for itself. Returns None when it chooses silence."""
    text, call = ask_chat_model(
        instructions=f"You are {agent}\n"
                     f"You know: {agent_knows or 'nothing beyond the conversation'}\n"
                     "Reply to the conversation briefly, as a helpful teammate. "
                     "If you have nothing worth adding, reply with exactly SILENT. "
                     "Stay silent for social chat, questions already answered, "
                     "or questions for someone else.",
        conversation=conversation,
    )
    return (None if text.upper().startswith("SILENT") else text), call
