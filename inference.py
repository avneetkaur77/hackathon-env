import os
import sys
import traceback
from server.hackathon_env_environment import HackathonEnvironment
from server.models import HackathonAction

# =========================
# MAIN
# =========================
def main():
    # --- Strict env vars ---
    try:
        API_BASE_URL = os.environ["API_BASE_URL"]
        API_KEY = os.environ["API_KEY"]
    except KeyError as ke:
        print(f"[FATAL] Missing required env var: {ke}", flush=True)
        sys.exit(1)

    # --- Import LLM inside main ---
    try:
        from openai import OpenAI
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        print("[CLIENT INIT] Success", flush=True)
    except Exception as e:
        print("[CLIENT INIT ERROR]:", str(e), flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)

    # --- Initialize env ---
    try:
        env = HackathonEnvironment()
    except Exception as e:
        print("[ENV INIT ERROR]:", str(e), flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)

    # --- Run episodes ---
    for _ in range(3):
        obs = env.reset()
        for step in range(3):
            # LLM call inside loop
            ticket_text = getattr(obs, "ticket_text", "") or ""
            try:
                res = client.responses.create(
                    model=os.environ.get("MODEL_NAME") or "gpt-3.5-turbo",
                    input=f"Classify into billing, refund, or replacement: {ticket_text}"
                )
                output = res.output[0].content[0].text.lower()
            except Exception as e:
                print("[LLM ERROR]:", str(e), flush=True)
                output = ""

            if "billing" in output:
                category, action, policy = "billing", "escalate", "standard"
            elif "refund" in output or "delay" in output:
                category, action, policy = "refund", "process_refund", "standard"
            else:
                category, action, policy = "replacement", "process_replacement", "standard"

            # Action for step
            act_type = ["classify", "investigate", action][min(step, 2)]
            act = HackathonAction(category=category, policy=policy, type=act_type, response="")
            obs = env.step(act)

    print("[END] task=boot", flush=True)

# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    main()