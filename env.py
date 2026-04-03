import random
from models import Observation

USER_STATES = ["studying", "sleeping", "free_time"]
NOTIFICATIONS = ["social", "work", "urgent"]

class NotificationEnv:

    def __init__(self):
        self.history = []
        self.current_state = None

    def reset(self):
        self.history = []
        self.current_state = self._generate_state()
        return self.current_state

    def state(self):
        return self.current_state

    def _generate_state(self):
        return Observation(
            user_state=random.choice(USER_STATES),
            notification_type=random.choice(NOTIFICATIONS),
            history=self.history[-3:]
        )

    def step(self, action):
        reward = self._calculate_reward(self.current_state, action)

        self.history.append(f"{self.current_state.notification_type}:{action}")

        done = len(self.history) >= 10

        self.current_state = self._generate_state()

        return self.current_state, reward, done, {}

    def _calculate_reward(self, state, action):
        reward = 0

        if state.user_state == "studying":
            if state.notification_type == "social":
                if action == "mute":
                    reward += 10
                elif action == "delay":
                    reward += 2
                else:
                    reward -= 10

            elif state.notification_type == "work":
                if action == "delay":
                    reward += 6
                elif action == "show_now":
                    reward += 3
                else:
                    reward -= 5

            elif state.notification_type == "urgent":
                if action == "show_now":
                    reward += 10
                else:
                    reward -= 8

        elif state.user_state == "sleeping":
            if action == "show_now":
                reward -= 10
            elif action == "mute":
                reward += 5

        elif state.user_state == "free_time":
            if action == "show_now":
                reward += 5
            elif action == "delay":
                reward += 2

        return reward