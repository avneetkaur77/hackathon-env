from fastapi import FastAPI, HTTPException
from server.models import HackathonAction
from server.hackathon_env_environment import HackathonEnvironment
import uvicorn

import threading
import subprocess

app = FastAPI()

# ✅ GLOBAL ENV
env = HackathonEnvironment()

# 🔥 RUN inference.py in background
def run_inference():
    try:
        subprocess.run(["python", "inference.py"], check=True)
    except Exception as e:
        print("[INFERENCE ERROR]:", str(e), flush=True)


@app.get("/")
def root():
    return {"message": "Hackathon Env Running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/reset")
@app.post("/reset")
def reset():
    try:
        obs = env.reset()
        return {
            "observation": {
                "ticket_text": obs.ticket_text,
                "metadata": obs.metadata
            },
            "reward": float(obs.reward),
            "done": bool(obs.done),
            "info": {}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(action: HackathonAction):
    try:
        obs = env.step(action)   # ✅ NO reset here
        return {
            "observation": {
                "ticket_text": obs.ticket_text,
                "metadata": obs.metadata
            },
            "reward": float(obs.reward),
            "done": bool(obs.done),
            "info": {}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    return {
        "episode_id": env.state.episode_id,
        "step_count": env.state.step_count
    }


# ✅ IMPORTANT: KEEP THIS FOR MODULE RUN
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()