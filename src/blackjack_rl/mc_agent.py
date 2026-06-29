from __future__ import annotations

from collections import defaultdict
from random import Random

from .env import State


class MonteCarloAgent:
    """First-visit Monte Carlo control for Blackjack."""

    def __init__(self, gamma: float = 1.0, seed: int = 21):
        self.gamma = gamma
        self.rng = Random(seed)
        self.q: dict[tuple[State, str], float] = defaultdict(float)
        self.visits: dict[tuple[State, str], int] = defaultdict(int)

    def choose_action(
        self,
        state: State,
        legal_actions: tuple[str, ...],
        epsilon: float,
    ) -> str:
        if self.rng.random() < epsilon:
            return self.rng.choice(legal_actions)
        return self.best_action(state, legal_actions)

    def best_action(self, state: State, legal_actions: tuple[str, ...]) -> str:
        return max(legal_actions, key=lambda action: self.q[(state, action)])

    def update(self, trajectory: list[tuple[State, str, float]]) -> None:
        """Update Q-values from one completed episode."""

        total_return = 0.0
        visited: set[tuple[State, str]] = set()

        for state, action, reward in reversed(trajectory):
            total_return = reward + self.gamma * total_return
            state_action = (state, action)

            if state_action in visited:
                continue

            visited.add(state_action)
            self.visits[state_action] += 1
            count = self.visits[state_action]
            old_value = self.q[state_action]
            self.q[state_action] = old_value + (total_return - old_value) / count
