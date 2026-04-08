import os
from openai import OpenAI
from server.hackathon_env_environment import HackathonEnvironment
from server.models import HackathonAction

client = None
MODEL_NAME = None


# =========================
# INIT CLIENT (FINAL SAFE)
# =========================
def init_client():
    global client, MODEL_NAME

    base_url = os.getenv("API_BASE_URL")
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME")

    if not base_url or not api_key:
        raise ValueError("❌ API_BASE_URL or API key missing")

    # ✅ fallback instead of crash
    if not MODEL_NAME:
        print("[WARNING] MODEL_NAME missing → using fallback")
        MODEL_NAME = "gpt-3.5-turbo"

    print("[DEBUG] BASE_URL:", base_url)
    print("[DEBUG] MODEL_NAME:", MODEL_NAME)

    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )


# =========================
# AGENT (SAFE)
# =========================
def intelligent_agent(observation):
    global client, MODEL_NAME

    ticket = observation.metadata or {}
    text = (ticket.get("text") or observation.ticket_text or "").lower()

    output = ""

    # ✅ LLM call with safety
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": f"Classify into billing, refund, or replacement: {text}"
                }
            ]
        )

        output = res.choices[0].message.content.lower()
        print("[LLM OUTPUT]:", output)

    except Exception as e:
        print("[LLM ERROR]:", str(e))

    # ✅ fallback logic (ensures progress)
    if "billing" in output:
        category, action = "billing", "escalate"
    elif "refund" in output or "delay" in output:
        category, action = "refund", "process_refund"
    else:
        category, action = "replacement", "process_replacement"

    return category, action, "Handled via LLM", "standard"


# =========================
# EPISODE RUNNER
# =========================
MAX_STEPS = 3

def run_episode(env):
    obs = env.reset()

    for step in range(1, MAX_STEPS + 1):
        category, action, response, policy = intelligent_agent(obs)

        if step == 1:
            act = HackathonAction(
                category=category,
                policy=policy,
                type="classify",
                response=""
            )
        elif step == 2:
            act = HackathonAction(
                category=category,
                policy=policy,
                type="investigate",
                response=""
            )
        else:
            act = HackathonAction(
                category=category,
                policy=policy,
                type=action,
                response=response
            )

        obs = env.step(act)

        if obs.done:
            break

    return getattr(obs, "reward", 0)


# =========================
# MAIN
# =========================
def main():
    init_client()

    env = HackathonEnvironment()
    scores = []

    for _ in range(3):
        try:
            scores.append(run_episode(env))
        except Exception as e:
            print("[EPISODE ERROR]:", str(e))
            scores.append(0)

    avg_score = sum(scores) / len(scores)
    print("AVG:", avg_score)


if __name__ == "__main__":
    main()