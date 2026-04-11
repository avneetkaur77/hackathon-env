---
title: Hackathon Env
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: server/app.py
pinned: false
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
- Uses LLM via proxy for intelligent decision-making
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

---

## API Endpoints

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
