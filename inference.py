import os
import requests
from openai import OpenAI

# 1. Setup Environment (Prevents Exception error)
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY")

if not API_KEY or not API_BASE_URL:
    raise ValueError("Missing API credentials")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
SERVER_URL = "http://localhost:7860"

def run_baseline():
    for task_id in ["easy", "medium", "hard"]:
        session_id = f"test_{task_id}"
        
        # RESET via API
        resp = requests.post(f"{SERVER_URL}/reset?task_id={task_id}&session_id={session_id}").json()
        obs = resp["observation"]
        done = False
        
        while not done:
            # 🔥 MANDATORY API CALL
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": f"Ticket: {obs['ticket_text']}. Solve it."}]
            )
            llm_text = completion.choices[0].message.content.lower()

            # MAP LLM TO ACTION (Based on your Agent logic)
            action = {
                "category": "billing" if "billing" in llm_text else "refund" if "refund" in llm_text else "replacement",
                "type": "investigate" if "investigate" in llm_text else "escalate",
                "response": "Sorry, we understand your issue. Priority refund.",
                "policy": "priority"
            }

            # STEP via API
            step_resp = requests.post(f"{SERVER_URL}/step?session_id={session_id}", json=action).json()
            obs = step_resp["observation"]
            done = step_resp["done"]
            print(f"Task {task_id} Reward: {step_resp['reward']}")

if __name__ == "__main__":
    run_baseline()