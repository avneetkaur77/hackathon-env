from fastapi import FastAPI, HTTPException
from server.models import HackathonAction
from server.hackathon_env_environment import HackathonEnvironment
import uvicorn

app = FastAPI()

# ✅ create fresh env per request (IMPORTANT)
def get_env():
    return HackathonEnvironment()


@app.get("/")
def root():
    return {"message": "Hackathon Env Running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ✅ RESET (GET + POST)
@app.get("/reset")
@app.post("/reset")
def reset():
    try:
        env = get_env()
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
        env = get_env()  # stateless (validator-friendly)
        env.reset()      # ensure valid state

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
        env = get_env()
        return {
            "episode_id": env.state.episode_id,
            "step_count": env.state.step_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ✅ REQUIRED for HF
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()