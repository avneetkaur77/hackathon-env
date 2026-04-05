<<<<<<< HEAD
---
title: Hackathon Env
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
---

# Adaptive Customer Support Decision Engine (OpenEnv)

## Overview

This project implements a **Reinforcement Learning (RL) environment** for customer support automation.

It simulates real-world support workflows where an AI agent:

- Understands customer issues  
- Classifies problem types  
- Selects appropriate policies  
- Executes correct actions  
- Generates empathetic responses  

The system is designed as a **deterministic and explainable decision engine**, ensuring **consistent and reproducible evaluation performance**.

---

## Problem Statement

Customer support requires structured multi-step decision-making:

- Issue classification  
- Policy selection  
- Context-aware prioritization  
- Consistent reasoning  

This project models the workflow as a **multi-step RL environment** for systematic evaluation.

---

## Key Design Principles

- Deterministic decision logic  
- Explainable reasoning  
- No external API dependency  
- Stable evaluation performance  

### Benefits

- Fully reproducible  
- No randomness  
- Lightweight and fast  
- Transparent decisions  

---

## Environment Design

### Observation Space

```json
{
  "ticket_text": "Customer complaint",
  "step": "current step",
  "history": "previous actions"
}
```

### Action Space

```json
{
  "category": "refund | replacement | billing",
  "policy": "standard | priority",
  "type": "classify | process_refund | process_replacement | escalate",
  "response": "natural language response"
}
```

---

## Multi-Step Decision Process

Each ticket follows:

1. Classification  
2. Investigation
3. Resolution

---

## Task Complexity Levels

- Easy → Clear refund or replacement requests  
- Medium → Delivery delays or ambiguous issues  
- Hard → Billing problems or complex scenarios  

---

## Reward Function

| Component            | Reward |
|---------------------|--------|
| Correct category     | +0.3   |
| Correct action       | +0.5   |
| Empathy              | +0.2   |
| Priority handling    | +0.1   |
| Personalization      | +0.2   |

Final reward is normalized between 0.0 and 1.0.

---

## Intelligent Agent Design

The agent is implemented as a deterministic decision engine combining:

- Keyword-based classification
- Context-based severity detection  
- Policy-based decision making 
- Structured response generation 

---

## Severity Modeling

Tickets are categorized into:

- Low  
- Medium  
- High  

Based on:

- Delay indicators  
- Customer sentiment  
- Urgency-related keywords  

---

## Policy Selection

- High severity → Priority policy  
- Otherwise → Standard policy  

---

## Explainability

Every decision made by the agent is fully interpretable and includes reasoning.

### Example Output

```
Final Decision
Category   : refund
Action     : process_refund
Policy     : priority
Confidence : 0.9
```

Response:

We are sorry for the inconvenience. We understand your concern. This case has been marked as priority. Since it has been 12 days, we will process your refund.

=======
# Adaptive Customer Support Decision Engine (OpenEnv)

## Overview
This project implements a deterministic Reinforcement Learning (RL) environment for customer support automation.

It simulates structured workflows including:
- Issue classification
- Policy selection
- Resolution handling

The system is fully deterministic, explainable, and reproducible.

---

## Features
- Multi-step decision process
- Deterministic outputs (no randomness)
- Reward-based evaluation
- Lightweight and fast
- OpenEnv compatible
>>>>>>> 3921579932e783ffda6cf62ab1f041fad24bd0a0

---

## API Endpoints

<<<<<<< HEAD
| Endpoint   | Description                     |
|----------|---------------------------------|
| /reset  | Start new ticket        |
| /step   | Execute action        |
| /state  | Get current state   |

---

## Performance

- Average Score: ~0.9
- Deterministic outputs
- Stable across runs  
- No external dependencies
---
## Project Structure
hackathon_env/
│── server/
│   ├── app.py
│   ├── models.py
│   ├── hackathon_env_environment.py
│
│── inference.py
│── Dockerfile
│── openenv.yaml
│── requirements.txt
│── README.md


## Setup Instructions

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Server

```bash
uvicorn server.app:app --reload
```

### Run Agent

```bash
python inference.py
```

---

## Optional Environment Variables

```bash
API_BASE_URL=local
MODEL_NAME=rule-based-agent
HF_TOKEN=dummy
```

---

## Docker

```bash
docker build -t hackathon-env .
docker run -p 7860:7860 hackathon-env
```

---

## HuggingFace Deployment
```bash
openenv push --repo-id <your-username>/hackathon-env
```
### After Deployment
https://<your-username>-hackathon-env.hf.space/reset

---

## Compliance

- Multi-step RL environment  
- Reward-based evaluation  
- Deterministic agent 
- Docker-compatible 
- OpenEnv API compliant

---

## Why This Approach Stands Out

- No API dependency
- Fully reproducible
- Explainable decisions
- Stable evaluation performance

This makes it ideal for evaluation-driven environments like hackathons.

---

## Determinism

- Fixed tasks
- No randomness
- Reproducible scoring


## Conclusion

This project demonstrates how customer support workflows can be modeled as a structured RL problem using deterministic decision-making.

It provides a reliable, interpretable, and evaluation-friendly alternative to API-based solutions.

=======
- `/reset` → Start a new ticket
- `/step` → Take an action
- `/state` → Get current state

---

## Example Output

```json
{
  "category": "refund",
  "action": "process_refund",
  "policy": "priority",
  "confidence": 0.9
}
Live Demo (HuggingFace Space)

👉 https://avneet77-hackathon-env.hf.space

Tech Stack-
Python
FastAPI
Docker
OpenEnv
Run Locally-
pip install -r requirements.txt
uvicorn server.app:app --reload
Author

Avneet Kaur
>>>>>>> 3921579932e783ffda6cf62ab1f041fad24bd0a0
