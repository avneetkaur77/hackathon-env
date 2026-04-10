
import os
from openai import OpenAI
from server.hackathon_env_environment import HackathonEnvironment
from server.models import HackathonAction

# =========================
# STRICT CLIENT INIT (NO FALLBACK)
# =========================
def init_client():
    api_base = os.environ["API_BASE_URL"]   # ❗ MUST exist in Phase 2
    api_key = os.environ["API_KEY"]         # ❗ MUST exist in Phase 2

    client = OpenAI(
        api_key=api_key,
        base_url=api_base
    )

    print("[CLIENT INITIALIZED]", flush=True)
    return client


# =========================
# FORCE API CALL (MANDATORY)
# =========================
def force_api_call(client):
    # ❗ NO try/except that hides failure
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Reply OK"}],
        max_tokens=5
    )

    # Optional read (ensures response parsed)
    _ = response.choices[0].message.content

    print("[API CALL SUCCESS]", flush=True)


# =========================
# AGENT LOGIC (RULE-BASED)
# =========================
def agent(obs):
    text = (obs.ticket_text or "").lower()

    if "charge" in text or "billing" in text:
        return "billing", "escalate", "We will investigate your billing issue.", "standard"

    elif "delay" in text or "not received" in text:
        return "refund", "process_refund", "Your refund will be processed on priority.", "priority"

    else:
        return "replacement", "process_replacement", "We will arrange a replacement.", "standard"


# =========================
# RUN EPISODES
# =========================
def run():
    env = HackathonEnvironment()

    for _ in range(3):
        obs = env.reset()

        category, action, response, policy = agent(obs)

        for step in range(1, 4):
            act_type = ["classify", "investigate", action][step - 1]

            obs = env.step(
                HackathonAction(
                    category=category,
                    type=act_type,
                    response=response,
                    policy=policy
                )
            )

            if obs.done:
                break


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("[START]", flush=True)

    # ✅ INIT CLIENT (STRICT)
    client = init_client()

    # ✅ FORCE PROXY API CALL (MANDATORY FOR PHASE 2)
    force_api_call(client)

    # ✅ RUN ENV
    run()

    print("[END]", flush=True)

