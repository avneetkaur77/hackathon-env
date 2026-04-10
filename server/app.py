from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import subprocess
import uvicorn

from server.models import HackathonAction
from server.hackathon_env_environment import HackathonEnvironment


# =========================
# 🔥 STARTUP FIX (CRITICAL)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run inference BEFORE server starts accepting requests
    try:
        print("[STARTING INFERENCE]", flush=True)
        subprocess.run(["python", "inference.py"], check=True)
        print("[INFERENCE COMPLETED]", flush=True)
    except Exception as e:
        print("[INFERENCE ERROR]:", str(e), flush=True)

    yield


app = FastAPI(lifespan=lifespan)


# =========================
# ✅ GLOBAL ENV (IMPORTANT)
# =========================
env = HackathonEnvironment()


# =========================
# ROUTES
# =========================
@app.get("/")
def root():
    return {"message": "Hackathon Env Running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ✅ RESET
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


# ✅ STEP
@app.post("/step")
def step(action: HackathonAction):
    try:
        obs = env.step(action)

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


# ✅ STATE
@app.get("/state")
def state():
    try:
        return {
            "episode_id": env.state.episode_id,
            "step_count": env.state.step_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# ENTRYPOINT
# =========================
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()