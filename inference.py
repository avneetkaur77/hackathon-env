import os
import textwrap
from openai import OpenAI
from server.hackathon_env_environment import HackathonEnvironment
from server.models import HackathonAction

# 1. STRICT ENV VARS (Requirement #2 & #3)
# Do not use .get() with a fallback. Let it raise KeyError if missing.
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

# Initialize Client
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def run_task(env, task_name):
    print(f"[START] task={task_name} env=hackathon model={MODEL_NAME}", flush=True)
    
    obs = env.reset()
    rewards = []
    
    # 2. LLM CALL INSIDE THE EPISODE (Requirement #1)
    # No try/except here that returns a dummy string (Requirement #4)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Classify: billing, refund, or replacement."},
            {"role": "user", "content": obs.ticket_text}
        ]
    )
    classification = completion.choices[0].message.content.lower()

    for step_idx in range(1, 4):
        # Logic to pick action based on LLM classification
        if step_idx == 1:
            category = "billing" if "billing" in classification else "refund"
            act = HackathonAction(category=category, type="classify")
        elif step_idx == 2:
            act = HackathonAction(type="investigate")
        else:
            act = HackathonAction(type="escalate", response="I understand the delay.")

        obs = env.step(act)
        
        # 3. MANDATORY STDOUT LOGGING
        reward = getattr(obs, "reward", 0.0)
        done = getattr(obs, "done", False)
        rewards.append(reward)
        
        print(f"[STEP] step={step_idx} action={act.type} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
        if done: break

    # 4. FINAL LOG
    score = sum(rewards)
    print(f"[END] success={str(score > 0.5).lower()} steps={step_idx} score={score:.2f} rewards={','.join(map(str, rewards))}", flush=True)

def main():
    env = HackathonEnvironment()
    # Run the 3 tasks defined in your environment
    for t in ["task1", "task2", "task3"]:
        run_task(env, t)

if __name__ == "__main__":
    main()