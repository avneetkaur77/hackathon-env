import os
from server.hackathon_env_environment import HackathonEnvironment
from server.models import HackathonAction

# =========================
# INTELLIGENT AGENT
# =========================
def intelligent_agent(obs, client, model_name):
    ticket_text = getattr(obs, "ticket_text", "") or ""

    # ✅ LLM call inside loop
    res = client.responses.create(
        model=model_name,
        input=f"Classify into billing, refund, or replacement: {ticket_text}"
    )
    output = res.output[0].content[0].text.lower()

    if "billing" in output:
        return "billing", "escalate", "standard"
    elif "refund" in output or "delay" in output:
        return "refund", "process_refund", "standard"
    else:
        return "replacement", "process_replacement", "standard"

# =========================
# RUN EPISODE
# =========================
def run_episode(env, client, model_name):
    obs = env.reset()
    total_reward, steps = 0, 0

    for step in range(3):
        category, action, policy = intelligent_agent(obs, client, model_name)
        act_type = ["classify", "investigate", action][min(step, 2)]
        act = HackathonAction(category=category, policy=policy, type=act_type, response="")
        obs = env.step(act)

        reward = getattr(obs, "reward", 0)
        total_reward += reward
        steps += 1

        if getattr(obs, "done", False):
            break

    return total_reward, steps

# =========================
# MAIN
# =========================
def main():
    env = HackathonEnvironment()
    client = env.client  # ✅ Use the environment-injected client
    model_name = os.environ.get("MODEL_NAME") or "gpt-3.5-turbo"

    total_score = 0
    total_steps = 0
    for _ in range(3):
        reward, steps = run_episode(env, client, model_name)
        total_score += reward
        total_steps += steps

    print(f"[END] task=boot score={total_score} steps={total_steps}", flush=True)

# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    main()