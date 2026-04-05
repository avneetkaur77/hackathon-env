from uuid import uuid4
from dataclasses import dataclass

from server.models import HackathonAction, HackathonObservation
# ✅ Local State (instead of openenv)
@dataclass
class State:
    episode_id: str
    step_count: int

class HackathonEnvironment:

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.max_steps = 3
        self.current_ticket = None

        # ✅ FIXED DATASET (DETERMINISTIC)
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

        # ✅ deterministic cycling
        self.current_ticket = self.tasks[self.task_index]
        self.task_index = (self.task_index + 1) % len(self.tasks)

        return HackathonObservation(
            ticket_text=self.current_ticket["text"],
            reward=0.0,
            done=False,
            metadata=self.current_ticket
        )

    def step(self, action: HackathonAction):
        self._state.step_count += 1
        step = self._state.step_count

        gt = self.current_ticket
        reward = 0.0
        response = (action.response or "").lower()

        # =========================
        # STEP 1: CATEGORY (HIGH IMPACT)
        # =========================
        if step == 1:
            if action.category == gt["category"]:
                reward = 1.0
            else:
                reward = 0.0

        # =========================
        # STEP 2: INVESTIGATION
        # =========================
        elif step == 2:
            if action.type == "investigate":
                reward = 1.0
            else:
                reward = 0.0

        # =========================
        # STEP 3: FINAL RESPONSE QUALITY
        # =========================
        elif step == 3:

            score = 0.0

            # correct action mapping
            if gt["category"] == "refund" and action.type == "process_refund":
                score += 0.3
            elif gt["category"] == "replacement" and action.type == "process_replacement":
                score += 0.3
            elif gt["category"] == "billing" and action.type == "escalate":
                score += 0.3

            # empathy
            if "sorry" in response:
                score += 0.2

            # understanding
            if "understand" in response:
                score += 0.2

            # personalization
            if str(gt["days"]) in response:
                score += 0.15

            # urgency handling
            if gt["is_urgent"] and "priority" in response:
                score += 0.15

            reward = min(1.0, score)

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