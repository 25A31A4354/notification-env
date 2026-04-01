# Smart Notification Manager AI

## Problem
Modern users are overwhelmed with notifications. This environment simulates how an AI agent can manage notifications intelligently based on user context.

## Objective
Train an AI to decide whether to:
- show notification
- delay it
- mute it

## User States
- studying
- sleeping
- free_time

## Notification Types
- social
- work
- urgent

## Actions
- show_now
- delay
- mute

## Reward Logic
- +10 → correct decision
- +5 → partially correct
- -5 → wrong decision

## Tasks
- Easy: simple decisions
- Medium: mixed scenarios
- Hard: complex situations

## How to Run

```bash
python3 inference.py