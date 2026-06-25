from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .agent import QLearningAgent
from .env import BlackjackEnv, State


def play_training_hand(env: BlackjackEnv, agent: QLearningAgent, epsilon: float) -> float:
    """Play one hand while learning from every action."""

    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        legal_actions = env.legal_actions()
        action = agent.choose_action(state, legal_actions, epsilon)

        next_state, reward, done = env.step(action)
        next_legal_actions = () if next_state is None else env.legal_actions()

        agent.update(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            next_legal_actions=next_legal_actions,
        )

        total_reward += reward
        if next_state is not None:
            state = next_state

    return total_reward


def train(episodes: int, seed: int) -> tuple[QLearningAgent, list[dict[str, float | int]]]:
    """Train the agent and keep progress logs."""

    env = BlackjackEnv(seed=seed)
    agent = QLearningAgent(seed=seed + 1)
    logs: list[dict[str, float | int]] = []
    window_profit = 0.0
    log_every = max(1, episodes // 20)

    for episode in range(1, episodes + 1):
        # At first the agent explores a lot. Later it trusts what it learned.
        epsilon = max(0.02, 0.30 * (1 - episode / episodes))
        window_profit += play_training_hand(env, agent, epsilon)

        if episode % log_every == 0:
            logs.append(
                {
                    "episode": episode,
                    "epsilon": round(epsilon, 4),
                    "avg_profit_last_window": round(window_profit / log_every, 5),
                    "known_state_actions": len(agent.q),
                }
            )
            window_profit = 0.0

    return agent, logs


def evaluate(agent: QLearningAgent, hands: int, seed: int) -> dict[str, float | int]:
    """Play hands using only the best learned action."""

    env = BlackjackEnv(seed=seed)
    profit = 0.0
    wins = 0
    losses = 0
    pushes = 0

    for _ in range(hands):
        state = env.reset()
        done = False
        hand_profit = 0.0

        while not done:
            action = agent.best_action(state, env.legal_actions())
            next_state, reward, done = env.step(action)
            hand_profit += reward
            if next_state is not None:
                state = next_state

        profit += hand_profit
        if hand_profit > 0:
            wins += 1
        elif hand_profit < 0:
            losses += 1
        else:
            pushes += 1

    return {
        "hands": hands,
        "total_profit": round(profit, 4),
        "avg_profit_per_hand": round(profit / hands, 6),
        "win_rate": round(wins / hands, 6),
        "loss_rate": round(losses / hands, 6),
        "push_rate": round(pushes / hands, 6),
        "learned_state_actions": len(agent.q),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_policy(agent: QLearningAgent) -> None:
    """Print the learned policy for all dealer upcards."""

    dealer_cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    dealer_labels = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"]

    print("\nLearned policy, no usable ace:")
    print("player_total | " + " | ".join(f"{label:>5}" for label in dealer_labels))
    for player_total in range(12, 22):
        actions = []
        for dealer_upcard in dealer_cards:
            state = State(player_total, dealer_upcard, False)
            actions.append(agent.best_action(state, ("hit", "stand")))
        print(f"{player_total:>12} | " + " | ".join(f"{action:>5}" for action in actions))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a basic Blackjack Q-learning player.")
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--eval-hands", type=int, default=10_000)
    parser.add_argument("--out", type=Path, default=Path("results_basic"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    agent, logs = train(episodes=args.episodes, seed=args.seed)
    summary = evaluate(agent, hands=args.eval_hands, seed=args.seed + 999)

    write_csv(args.out / "training_log.csv", logs)
    write_csv(args.out / "evaluation_summary.csv", [summary])

    print(summary)
    print_policy(agent)
    print(f"\nWrote logs to: {args.out}")


if __name__ == "__main__":
    main()
