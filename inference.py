import os

from server.hackathon_env_environment import HackathonEnvironment
from server.models import HackathonAction

# ✅ KEEP THESE (required by evaluator)
API_BASE_URL = os.getenv("API_BASE_URL", "local")
MODEL_NAME = os.getenv("MODEL_NAME", "rule-based-agent")
HF_TOKEN = os.getenv("HF_TOKEN")

MAX_STEPS = 3


# =========================
# INTELLIGENT AGENT (UNCHANGED)
# =========================

def intelligent_agent(observation):
    ticket = observation.metadata

    text = (ticket.get("text") or observation.ticket_text).lower()
    days = ticket.get("days", 0)
    is_urgent = ticket.get("is_urgent", False)

    reasoning = []

    if any(w in text for w in ["charge", "charged", "billing", "payment", "invoice"]):
        category = "billing"
        reasoning.append("Billing detected")

    elif any(w in text for w in [
        "not received", "lost", "missing", "delay",
        "not delivered", "haven't received", "still haven't", "package"
    ]):
        category = "refund"
        reasoning.append("Delivery detected")

    elif any(w in text for w in [
        "broken", "damaged", "defective", "not working"
    ]):
        category = "replacement"
        reasoning.append("Product detected")

    elif any(w in text for w in ["idk", "something", "maybe", "issue", "??"]):

        if "working" in text:
            category = "replacement"
        elif "order" in text:
            category = "replacement"
        else:
            category = "replacement"

    else:
        category = "replacement"

    if category == "billing":
        action = "escalate"
    elif category == "refund":
        action = "process_refund"
    else:
        action = "process_replacement"

    policy = "priority" if (is_urgent or days >= 10) else "standard"

    response = "We are sorry for the inconvenience. "
    response += "We understand your concern. "

    if policy == "priority":
        response += "This case has been marked as priority. "

    if category == "refund":
        response += f"Since it has been {days} days, we will process your refund. "
    elif category == "replacement":
        response += "We will resolve this by arranging a replacement. "
    elif category == "billing":
        response += "We will investigate and resolve the billing issue immediately. "

    response += f"Our team will review, investigate, and resolve your issue regarding your order from {days} days ago."

    confidence = 0.9

    return category, action, response, policy, confidence, reasoning


# =========================
# RUN EPISODE (ONLY LOG FIX)
# =========================

def run_episode(env, episode_id):
    obs = env.reset()

    print(f"[STEP] episode={episode_id} step=0 observation='{obs.ticket_text}'")

    category, action, response, policy, confidence, reasoning = intelligent_agent(obs)

    for step in range(1, MAX_STEPS + 1):

        if step == 1:
            act = HackathonAction(category=category, policy=policy, type="classify", response="")
        elif step == 2:
            act = HackathonAction(category=category, policy=policy, type="investigate", response="")
        else:
            act = HackathonAction(category=category, policy=policy, type=action, response=response)

        obs = env.step(act)

        # ✅ REQUIRED FORMAT
        print(f"[STEP] episode={episode_id} step={step} reward={obs.reward}")

        if obs.done:
            break

    return obs.reward


# =========================
# MAIN (FIXED FORMAT)
# =========================

def main():
    print("[START]")

    env = HackathonEnvironment()
    scores = []

    for i in range(3):
        score = run_episode(env, i + 1)
        scores.append(score)

    print(f"[END] avg_score={round(sum(scores)/len(scores), 2)}")


if __name__ == "__main__":
    main()