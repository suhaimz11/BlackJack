from __future__ import annotations

import argparse
import csv
from pathlib import Path
from random import Random

from .env import BlackjackEnv
from .mc_agent import MonteCarloAgent
from .train_basic import (
    PracticeCell,
    basic_strategy_accuracy,
    double_strategy_accuracy,
    pair_split_accuracy,
    practice_cells,
    print_basic_strategy_accuracy,
    print_double_policy,
    print_policy,
    soft_strategy_accuracy,
)


def play_mc_hand(
    env: BlackjackEnv,
    agent: MonteCarloAgent,
    epsilon: float,
    practice_cell: PracticeCell | None = None,
) -> float:
    """Play one full hand and update with first-visit Monte Carlo."""

    if practice_cell is None:
        state = env.reset()
    else:
        kind, value, dealer_upcard = practice_cell
        if kind == "pair":
            state = env.reset_to_pair(value, dealer_upcard)
        elif kind in ("soft", "double_soft"):
            state = env.reset_to_soft_total(value, dealer_upcard)
        else:
            state = env.reset_to_hard_total(value, dealer_upcard)

    done = False
    trajectory = []
    total_reward = 0.0

    while not done:
        legal_actions = env.legal_actions()
        action = agent.choose_action(state, legal_actions, epsilon)
        next_state, reward, done = env.step(action)
        trajectory.append((state, action, reward))
        total_reward += reward
        if next_state is not None:
            state = next_state

    agent.update(trajectory)
    return total_reward


def train(
    episodes: int,
    seed: int,
    practice_ratio: float,
) -> tuple[MonteCarloAgent, list[dict[str, float | int | str]]]:
    env = BlackjackEnv(seed=seed)
    agent = MonteCarloAgent(seed=seed + 1)
    practice_rng = Random(seed + 2)
    cells = practice_cells()
    logs: list[dict[str, float | int | str]] = []
    window_profit = 0.0
    log_every = max(1, episodes // 20)

    for episode in range(1, episodes + 1):
        epsilon = max(0.01, 0.30 * (1 - episode / episodes))
        practice_cell = None
        if practice_rng.random() < practice_ratio:
            practice_cell = cells[(episode - 1) % len(cells)]

        window_profit += play_mc_hand(env, agent, epsilon, practice_cell)

        if episode % log_every == 0:
            logs.append(
                {
                    "algorithm": "first_visit_monte_carlo",
                    "episode": episode,
                    "epsilon": round(epsilon, 4),
                    "practice_ratio": practice_ratio,
                    "practice_cells": len(cells),
                    "avg_profit_last_window": round(window_profit / log_every, 5),
                    "known_state_actions": len(agent.q),
                }
            )
            window_profit = 0.0

    return agent, logs


def evaluate(agent: MonteCarloAgent, hands: int, seed: int) -> dict[str, float | int | str]:
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
        "algorithm": "first_visit_monte_carlo",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a basic Blackjack Monte Carlo player.")
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--eval-hands", type=int, default=20_000)
    parser.add_argument("--practice-ratio", type=float, default=0.8)
    parser.add_argument("--out", type=Path, default=Path("results_basic_mc"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    agent, logs = train(args.episodes, args.seed, args.practice_ratio)
    summary = evaluate(agent, args.eval_hands, args.seed + 999)

    hard_score = basic_strategy_accuracy(agent)
    soft_score = soft_strategy_accuracy(agent)
    double_score = double_strategy_accuracy(agent)
    split_score = pair_split_accuracy(agent)
    summary["practice_ratio"] = args.practice_ratio
    summary["hard_total_strategy_matches"] = hard_score["matches"]
    summary["hard_total_strategy_total"] = hard_score["total"]
    summary["hard_total_strategy_accuracy"] = hard_score["accuracy"]
    summary["soft_total_strategy_matches"] = soft_score["matches"]
    summary["soft_total_strategy_total"] = soft_score["total"]
    summary["soft_total_strategy_accuracy"] = soft_score["accuracy"]
    summary["double_strategy_matches"] = double_score["matches"]
    summary["double_strategy_total"] = double_score["total"]
    summary["double_strategy_accuracy"] = double_score["accuracy"]
    summary["pair_split_strategy_matches"] = split_score["matches"]
    summary["pair_split_strategy_total"] = split_score["total"]
    summary["pair_split_strategy_accuracy"] = split_score["accuracy"]

    write_csv(args.out / "training_log.csv", logs)
    write_csv(args.out / "evaluation_summary.csv", [summary])

    print(summary)
    print_policy(agent)
    print_double_policy(agent)
    print_basic_strategy_accuracy(agent)
    print(f"\nWrote logs to: {args.out}")


if __name__ == "__main__":
    main()
