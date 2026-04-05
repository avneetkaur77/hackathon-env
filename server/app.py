from fastapi import FastAPI
from server.models import HackathonAction
from server.hackathon_env_environment import HackathonEnvironment
import uvicorn  # <-- needed to run app in main()

app = FastAPI()
env = HackathonEnvironment()

@app.get("/health")
def health():
    return {"status": "ok"}

# ✅ Support both GET and POST for reset
@app.get("/reset")
@app.post("/reset")
def reset():
    obs = env.reset()
    return {
        "observation": {
            "ticket_text": obs.ticket_text,
            "metadata": obs.metadata
        },
        "reward": obs.reward,
        "done": obs.done,
        "info": {}
    }

@app.post("/step")
def step(action: HackathonAction):
    obs = env.step(action)
    return {
        "observation": {
            "ticket_text": obs.ticket_text,
            "metadata": obs.metadata
        },
        "reward": obs.reward,
        "done": obs.done,
        "info": {}
    }

@app.get("/state")
def state():
    return {
        "episode_id": env.state.episode_id,
        "step_count": env.state.step_count
    }

# ✅ Add main() for OpenEnv multi-mode / HF Spaces
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()