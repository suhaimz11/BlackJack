from __future__ import annotations

from collections import defaultdict
from random import Random

from .env import State


class QLearningAgent:
    """A small Q-learning agent for Blackjack."""

    def __init__(self, alpha: float = 0.02, gamma: float = 1.0, seed: int = 11):
        self.alpha = alpha
        self.gamma = gamma
        self.rng = Random(seed)

        # q[(state, action)] stores how good an action seems in a state.
        self.q: dict[tuple[State, str], float] = defaultdict(float)

    def choose_action(
        self,
        state: State,
        legal_actions: tuple[str, ...],
        epsilon: float,
    ) -> str:
        """Sometimes explore randomly, otherwise choose the best known action."""

        if self.rng.random() < epsilon:
            return self.rng.choice(legal_actions)

        return self.best_action(state, legal_actions)

    def best_action(self, state: State, legal_actions: tuple[str, ...]) -> str:
        """Choose the action with the highest Q-value."""

        return max(legal_actions, key=lambda action: self.q[(state, action)])

    def update(
        self,
        state: State,
        action: str,
        reward: float,
        next_state: State | None,
        next_legal_actions: tuple[str, ...],
    ) -> None:
        """Move the Q-value a little closer to the observed result."""

        old_value = self.q[(state, action)]

        if next_state is None:
            best_future_value = 0.0
        else:
            best_future_value = max(
                self.q[(next_state, next_action)]
                for next_action in next_legal_actions
            )

        target = reward + self.gamma * best_future_value
        self.q[(state, action)] = old_value + self.alpha * (target - old_value)
