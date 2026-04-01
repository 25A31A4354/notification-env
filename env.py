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

    def _generate_state(self):
        return Observation(
            user_state=random.choice(USER_STATES),
            notification_type=random.choice(NOTIFICATIONS),
            history=self.history[-3:]
        )

    def step(self, action):
        reward = self._calculate_reward(self.current_state, action)

        self.history.append(f"{self.current_state.notification_type}:{action}")

        done = len(self.history) >= 5

        self.current_state = self._generate_state()

        return self.current_state, reward, done, {}

    def _calculate_reward(self, state, action):
        if state.user_state in ["studying", "sleeping"]:
            if state.notification_type == "social" and action == "mute":
                return 10
            if state.notification_type == "work" and action == "delay":
                return 5
            if state.notification_type == "urgent" and action == "show_now":
                return 10

        elif state.user_state == "free_time":
            if state.notification_type == "social" and action == "show_now":
                return 5
            if state.notification_type == "urgent" and action == "show_now":
                return 10

        return -5