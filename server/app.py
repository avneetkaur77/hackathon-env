
from fastapi import FastAPI, HTTPException
from server.models import HackathonAction
from server.hackathon_env_environment import HackathonEnvironment
import uvicorn

app = FastAPI()

# ✅ GLOBAL ENV (IMPORTANT FIX)
env = HackathonEnvironment()


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
    global env
    try:
        env = HackathonEnvironment()  # fresh episode
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


# ✅ STEP (FIXED)
@app.post("/step")
def step(action: HackathonAction):
    global env
    try:
        obs = env.step(action)   # ❌ NO reset here

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
    global env
    try:
        return {
            "episode_id": env.state.episode_id,
            "step_count": env.state.step_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

