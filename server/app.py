import uvicorn
from fastapi import FastAPI, HTTPException
from server.models import HackathonAction
from server.hackathon_env_environment import HackathonEnvironment

app = FastAPI(title="Adaptive Customer Support Engine")

# This dictionary stores the state so Step 1, 2, and 3 stay linked
sessions = {}

@app.get("/")
def root():
    return {"message": "Customer Support Environment is Live"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ✅ FIXED: Added Reset endpoint (Validator hits this first)
@app.post("/reset")
def reset(task_id: str = "easy", session_id: str = "default"):
    try:
        env = HackathonEnvironment()
        # Start the specific task requested (easy, medium, or hard)
        obs = env.reset_to_task(task_id)
        sessions[session_id] = env 
        
        return {
            "observation": {
                "ticket_text": obs.ticket_text,
                "metadata": obs.metadata
            },
            "reward": 0.0,
            "done": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ FIXED: Step no longer calls reset() internally
@app.post("/step")
def step(action: HackathonAction, session_id: str = "default"):
    if session_id not in sessions:
        # If agent forgets to reset, we do it once for them
        sessions[session_id] = HackathonEnvironment()
        sessions[session_id].reset()
    
    env = sessions[session_id]
    obs = env.step(action)

    return {
        "observation": {
            "ticket_text": obs.ticket_text,
            "metadata": obs.metadata
        },
        "reward": float(obs.reward),
        "done": bool(obs.done)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)