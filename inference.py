from server.hackathon_env_environment import HackathonEnvironment
from server.models import HackathonAction

# =========================
# INTELLIGENT AGENT
# =========================
def intelligent_agent(obs, client, model_name):
    ticket_text = getattr(obs, "ticket_text", "") or ""

    # LLM call inside step loop
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
    for step in range(3):
        category, action, policy = intelligent_agent(obs, client, model_name)
        act_type = ["classify", "investigate", action][min(step, 2)]
        act = HackathonAction(category=category, policy=policy, type=act_type, response="")
        obs = env.step(act)

# =========================
# MAIN
# =========================
def main():
    env = HackathonEnvironment()
    # Use hackathon-provided client automatically injected in env
    client = env.get_client()
    model_name = os.environ.get("MODEL_NAME") or "gpt-3.5-turbo"

    for _ in range(3):
        run_episode(env, client, model_name)

if __name__ == "__main__":
    main()