---
title: Smart Notification Manager AI
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# Smart Notification Manager AI

## 🚀 Overview
This project simulates a real-world environment where an AI agent learns to manage notifications intelligently based on user context.

In today's world, constant notifications reduce productivity and focus. This environment trains AI to prioritize notifications in a human-like, context-aware manner.

---

## 🧠 Problem Statement
Users receive various types of notifications (social, work, urgent), but not all should be treated equally.

This environment helps an AI agent learn:
- When to show a notification
- When to delay it
- When to mute it completely

---

## ⚙️ Environment Design

### 👤 User States
- Studying
- Sleeping
- Free Time

### 🔔 Notification Types
- Social (Instagram, WhatsApp)
- Work (Emails, tasks)
- Urgent (Calls, emergencies)

### 🎯 Actions
- `show_now`
- `delay`
- `mute`

---

## 🧪 Tasks

- **Easy** → Clear decision-making scenarios  
- **Medium** → Mixed notification priorities  
- **Hard** → Complex real-world situations with ambiguity  

---

## 🎯 Reward Logic

| Scenario | Correct Action | Reward |
|---------|--------------|--------|
| Studying + Social | Mute | +10 |
| Studying + Work | Delay | +5 |
| Urgent Notifications | Show Now | +10 |
| Wrong Decision | Penalty | -5 |

The reward system encourages focus preservation and correct prioritization.

---

## 📊 Evaluation

The agent is evaluated using a scoring system between **0.0 and 1.0** based on total reward.

---

## ▶️ How to Run

```bash
python3 inference.py