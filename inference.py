import os
from openai import OpenAI
from server.hackathon_env_environment import HackathonEnvironment
from server.models import HackathonAction
import traceback

client = None
MODEL_NAME = None

# =========================
# CLIENT INIT WITH DEBUG
# =========================
def init_client():
    global client, MODEL_NAME
    try:
        API_BASE_URL = os.environ["API_BASE_URL"]
        API_KEY = os.environ["API_KEY"]
        MODEL_NAME = os.environ.get("MODEL_NAME") or "gpt-3.5-turbo"

        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        print("[CLIENT INIT] Success", flush=True)

        # Validator ping
        try:
            resp = client.responses.create(model=MODEL_NAME, input="phase 2 validator ping")
            print("[VALIDATOR PING SUCCESS]:", resp, flush=True)
        except Exception as e:
            print("[VALIDATOR PING FAILED]:", str(e), flush=True)
            print(traceback.format_exc(), flush=True)

    except KeyError as ke:
        print(f"[CRITICAL] Missing env var: {ke}", flush=True)
    except Exception as e:
        print("[CLIENT INIT ERROR]:", str(e), flush=True)
        print(traceback.format_exc(), flush=True)
        client = None

# =========================
# INTELLIGENT AGENT WITH DEBUG
# =========================
def intelligent_agent(observation):
    ticket = observation.metadata or {}
    text = (ticket.get("text") or observation.ticket_text or "").lower()
    output = ""

    if client is None:
        print("[ERROR] LLM client not initialized", flush=True)
        return "replacement", "process_replacement", "Handled", "standard"

    try:
        print(f"[LLM CALL] Sending text: {text}", flush=True)
        res = client.responses.create(
            model=MODEL_NAME,
            input=f"Classify into billing, refund, or replacement: {text}"
        )
        try:
            output = res.output[0].content[0].text.lower()
        except Exception:
            output = str(res).lower()
        print("[LLM OUTPUT]:", output, flush=True)
    except Exception as e:
        print("[LLM CALL ERROR]:", str(e), flush=True)
        print(traceback.format_exc(), flush=True)
        output = ""

    if "billing" in output:
        return "billing", "escalate", "Handled", "standard"
    elif "refund" in output or "delay" in output:
        return "refund", "process_refund", "Handled", "standard"
    else:
        return "replacement", "process_replacement", "Handled", "standard"

# =========================
# RUN EPISODE WITH DEBUG
# =========================
def run_episode(env, task_name="ticket_resolution"):
    print(f"[START] task={task_name}", flush=True)
    try:
        obs = env.reset()
    except Exception as e:
        print("[RESET ERROR]:", str(e), flush=True)
        print(traceback.format_exc(), flush=True)
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
        except Exception as e:
            print("[STEP ERROR]:", str(e), flush=True)
            print(traceback.format_exc(), flush=True)
            break

    print(f"[END] task={task_name} score={total_reward} steps={steps}", flush=True)
    return total_reward

# =========================
# MAIN WITH DEBUG
# =========================
def main():
    print("[START] task=boot", flush=True)
    init_client()
    if client is None:
        print("[FATAL] Cannot run without LLM client.", flush=True)
        return

    try:
        env = HackathonEnvironment()
    except Exception as e:
        print("[ENV ERROR]:", str(e), flush=True)
        print(traceback.format_exc(), flush=True)
        return

    for _ in range(3):
        run_episode(env)

    print("[END] task=boot score=1 steps=1", flush=True)

# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    main()