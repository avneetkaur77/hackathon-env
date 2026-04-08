import os
from openai import OpenAI
from server.hackathon_env_environment import HackathonEnvironment
from server.models import HackathonAction

client = None

def init_client():
    global client

    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"]
    )


MAX_STEPS = 3

def intelligent_agent(observation):
    global client

    ticket = observation.metadata or {}
    text = (ticket.get("text") or observation.ticket_text or "").lower()

    # ✅ STRICT LLM CALL (NO SILENT FAIL)
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 🔥 safer for proxy mapping
        messages=[
            {"role": "user", "content": f"Classify into billing, refund, or replacement: {text}"}
        ]
    )

    output = res.choices[0].message.content.lower()
    print("[LLM OUTPUT]:", output)

    # ✅ USE OUTPUT
    if "billing" in output:
        category, action = "billing", "escalate"
    elif "refund" in output or "delay" in output:
        category, action = "refund", "process_refund"
    else:
        category, action = "replacement", "process_replacement"

    policy = "standard"
    response_text = "Handled via LLM"

    return category, action, response_text, policy


def run_episode(env):
    obs = env.reset()

    for step in range(1, MAX_STEPS + 1):
        category, action, response, policy = intelligent_agent(obs)

        if step == 1:
            act = HackathonAction(category=category, policy=policy, type="classify", response="")
        elif step == 2:
            act = HackathonAction(category=category, policy=policy, type="investigate", response="")
        else:
            act = HackathonAction(category=category, policy=policy, type=action, response=response)

        obs = env.step(act)

        if obs.done:
            break

    return getattr(obs, "reward", 0)


def main():
    init_client()

    env = HackathonEnvironment()
    scores = []

    for _ in range(3):
        scores.append(run_episode(env))

    print("AVG:", sum(scores) / len(scores))


if __name__ == "__main__":
    main()