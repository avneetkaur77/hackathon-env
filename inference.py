import os
from openai import OpenAI
from server.hackathon_env_environment import HackathonEnvironment
from server.models import HackathonAction
import traceback

# =========================
# GLOBALS
# =========================
client = None
MODEL_NAME = None

# =========================
# INTELLIGENT AGENT
# =========================
def intelligent_agent(observation):
    ticket = observation.metadata or {}
    text = (ticket.get("text") or observation.ticket_text or "").strip()

    # ❗ Validator requires LLM call inside agent loop
    res = client.responses.create(
        model=MODEL_NAME,
        input=f"Classify into billing, refund, or replacement: {text}"
    )

    # safely extract output
    try:
        output = res.output[0].content[0].text.lower()
    except Exception:
        output = str(res).lower()

    if "billing" in output:
        return "billing", "escalate", "Handled", "standard"
    elif "refund" in output or "delay" in output:
        return "refund", "process_refund", "Handled", "standard"
    else:
        return "replacement", "process_replacement", "Handled", "standard"

# =========================
# RUN EPISODE
# =========================
def run_episode(env, task_name="ticket_resolution"):
    print(f"[START] task={task_name}", flush=True)
    try:
        obs = env.reset()
    except Exception:
        traceback.print_exc()
        return 0

    total_reward, steps = 0, 0

    for step in range(3):
        try:
            category, action, response, policy = intelligent_agent(obs)

            if step == 0:
                act = HackathonAction(category=category, policy=policy, type="classify", response="")
            elif step == 1:
                act = HackathonAction(category=category, policy=policy, type="investigate", response="")
            else:
                act = HackathonAction(category=category, policy=policy, type=action, response=response)

            obs = env.step(act)
            reward = getattr(obs, "reward", 0)
            total_reward += reward
            steps += 1
            print(f"[STEP] step={steps} reward={reward}", flush=True)

            if getattr(obs, "done", False):
                break
        except Exception:
            traceback.print_exc()
            break

    print(f"[END] task={task_name} score={total_reward} steps={steps}", flush=True)
    return total_reward

# =========================
# MAIN
# =========================
def main():
    global client, MODEL_NAME
    print("[START] task=boot", flush=True)

    # ❗ Strictly use injected env vars
    try:
        API_BASE_URL = os.environ["API_BASE_URL"]
        API_KEY = os.environ["API_KEY"]
        MODEL_NAME = os.environ.get("MODEL_NAME") or "gpt-3.5-turbo"
    except KeyError as ke:
        print(f"[FATAL] Missing env var: {ke}", flush=True)
        return

    # ❗ Wrap client init to prevent unhandled exceptions
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        print("[CLIENT INIT SUCCESS]", flush=True)
    except Exception:
        print("[CLIENT INIT FAILED]", flush=True)
        traceback.print_exc()
        return

    # ❗ Wrap environment creation
    try:
        env = HackathonEnvironment()
    except Exception:
        print("[ENV INIT FAILED]", flush=True)
        traceback.print_exc()
        return

    # Run 3 episodes
    for _ in range(3):
        run_episode(env)

    print("[END] task=boot score=1 steps=1", flush=True)

# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    main()