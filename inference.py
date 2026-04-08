import os
import textwrap
from openai import OpenAI
from server.hackathon_env_environment import HackathonEnvironment
from server.models import HackathonAction

# 1. STRICT ENV VARS
# Using os.environ[] ensures the script fails immediately if the 
# validator has not injected the keys, preventing "zombie" runs.
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

# Initialize OpenAI Client
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def run_task(env, task_name):
    # [START] is mandatory for scoring
    print(f"[START] task={task_name} env=hackathon_env model={MODEL_NAME}", flush=True)
    
    obs = env.reset()
    rewards = []
    steps_taken = 0
    
    # 2. LLM CALL INSIDE THE EPISODE
    # We do NOT use try/except here. If the proxy fails, we want the error visible.
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Classify this ticket: billing, refund, or replacement."},
            {"role": "user", "content": obs.ticket_text}
        ],
        temperature=0
    )
    classification = completion.choices[0].message.content.lower()

    for step_idx in range(1, 4):
        # Determine the action based on the step logic in your environment
        if step_idx == 1:
            category = "billing" if "billing" in classification else ("refund" if "refund" in classification else "replacement")
            act = HackathonAction(category=category, type="classify")
        elif step_idx == 2:
            act = HackathonAction(type="investigate")
        else:
            # Final Response Step
            final_type = "escalate" if "billing" in classification else ("process_refund" if "refund" in classification else "process_replacement")
            act = HackathonAction(type=final_type, response="I understand the situation and have processed the request.")

        # Step the environment
        obs = env.step(act)
        
        reward = getattr(obs, "reward", 0.0)
        done = getattr(obs, "done", False)
        rewards.append(reward)
        steps_taken = step_idx
        
        # [STEP] is mandatory
        print(f"[STEP] step={step_idx} action={act.type} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
        
        if done:
            break

    # 3. FINAL LOGGING
    total_score = sum(rewards)
    success = total_score > 0.5
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps_taken} score={total_score:.2f} rewards={rewards_str}", flush=True)

def main():
    # Initialize the local environment class
    env = HackathonEnvironment()
    
    # Run through the tasks defined in your environment.py
    # This ensures the agent tackles all 3 difficulty levels.
    for t_name in ["easy", "medium", "hard"]:
        run_task(env, t_name)

if __name__ == "__main__":
    main()