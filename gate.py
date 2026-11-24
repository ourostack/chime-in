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

# Question 1, the gate: how valuable would a reply be right now?
# The decision model gives a score from 0 (the first level) to 3 (the last).
HOW_VALUABLE = [
    "None: nothing to add; the chat is social, already answered, for someone else, "
    "or the agent knows nothing useful about it",
    "Low: mildly helpful, people would not miss it",
    "Medium: saves someone real time or effort",
    "High: people need this now and would be worse off without it",
]

# The rule: the agent replies only if the reply is worth at least this much.
REPLY_IF_VALUE_AT_LEAST = 1.5

# Question 2: if the agent does reply, what kind of reply?
# This doesn't decide whether to speak; it tells the LLM what to write.
KIND_OF_REPLY = {
    "answer_question": "Answer an open question from what the agent knows",
    "share_context": "Share something relevant the people talking don't seem to have",
    "correct_mistake": "Correct something someone stated that the agent knows is wrong",
    "do_task": "Do something someone asked the agent itself to do",
}


@dataclass
class Decision:
    reply: bool                       # the verdict: should the agent chime in?
    value: float                      # question 1: the score, 0 to 3
    kind: str                         # question 2: the kind of reply the model picked
    kind_probabilities: dict[str, float]  # question 2: the probability of every kind
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
            "how_valuable": score_on_a_scale(
                HOW_VALUABLE,
                "The agent is a quiet member of this conversation. If it replied "
                "to the latest messages now, how valuable would that reply be?"),
            "kind_of_reply": choose_one_of(
                KIND_OF_REPLY,
                "If the agent did reply now, what kind of reply would help most?"),
        },
    )

    value = answers["how_valuable"]["score"]
    kind = answers["kind_of_reply"]

    return Decision(
        reply=value >= REPLY_IF_VALUE_AT_LEAST,
        value=value,
        kind=kind["choice"],
        kind_probabilities=kind["probabilities"],
        call=call,
    )


def write_reply(agent: str, conversation: list[dict], agent_knows: str = "",
                kind: str = "") -> tuple[str, Call]:
    """Only after the gate says yes: the full LLM writes the kind of reply the gate picked."""
    return ask_chat_model(
        instructions=f"You are {agent}\n"
                     f"You know: {agent_knows or 'nothing beyond the conversation'}\n"
                     + (f"Your reply should: {KIND_OF_REPLY[kind]}.\n" if kind in KIND_OF_REPLY else "")
                     + "Reply to the conversation briefly, as a helpful teammate.",
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
