from uuid import uuid4
from dataclasses import dataclass
from server.models import HackathonAction, HackathonObservation

# =========================
# UTILS
# =========================
def clamp_reward(value: float) -> float:
    return max(0.0, min(1.0, value))

# =========================
# STATE
# =========================
@dataclass
class State:
    episode_id: str
    step_count: int

# =========================
# ENVIRONMENT
# =========================
class HackathonEnvironment:

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.max_steps = 3
        self.current_ticket = None

        self.tasks = [
            {
                "id": "easy",
                "text": "I was charged twice for my order",
                "category": "billing",
                "days": 2,
                "is_urgent": False
            },
            {
                "id": "medium",
                "text": "It's been 12 days and my package has not arrived",
                "category": "refund",
                "days": 12,
                "is_urgent": True
            },
            {
                "id": "hard",
                "text": "idk something wrong with my order not working??",
                "category": "replacement",
                "days": 5,
                "is_urgent": False
            }
        ]

        self.task_index = 0

    def reset(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.current_ticket = self.tasks[self.task_index]
        self.task_index = (self.task_index + 1) % len(self.tasks)

        return HackathonObservation(
            ticket_text=self.current_ticket["text"],
            reward=0.0,
            done=False,
            metadata=self.current_ticket
        )

    def step(self, action: HackathonAction):
        if self.current_ticket is None:
            self.reset()

        self._state.step_count += 1
        step = self._state.step_count

        gt = self.current_ticket
        reward = 0.0
        response = (action.response or "").lower()

        # STEP 1: CATEGORY
        if step == 1:
            reward = 1.0 if action.category == gt["category"] else 0.0

        # STEP 2: INVESTIGATION
        elif step == 2:
            reward = 1.0 if action.type == "investigate" else 0.0

        # STEP 3: FINAL RESPONSE
        elif step == 3:
            score = 0.0

            # category/action type matching
            if gt["category"] == "refund" and action.type == "process_refund":
                score += 0.3
            elif gt["category"] == "replacement" and action.type == "process_replacement":
                score += 0.3
            elif gt["category"] == "billing" and action.type == "escalate":
                score += 0.3

            # text-based bonus signals
            if "sorry" in response:
                score += 0.2
            if "understand" in response:
                score += 0.2
            if str(gt["days"]) in response:
                score += 0.15
            if gt["is_urgent"] and "priority" in response:
                score += 0.15

            reward = clamp_reward(min(1.0, score))

        done = step >= self.max_steps

        return HackathonObservation(
            ticket_text=gt["text"] if done else "Processing...",
            reward=round(reward, 2),
            done=done,
            metadata={"step": step, "task_id": gt["id"]}
        )

    @property
    def state(self):
        return self._state