from __future__ import annotations

import argparse
import csv
from pathlib import Path
from random import Random

from .agent import QLearningAgent
from .env import BlackjackEnv, State


DEALER_CARDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
DEALER_LABELS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"]

# Standard hard-total basic strategy for hit/stand only.
# A dealer ace is represented as 11 in the code.
HARD_TOTAL_BASIC_STRATEGY: dict[int, dict[int, str]] = {
    12: {2: "hit", 3: "hit", 4: "stand", 5: "stand", 6: "stand", 7: "hit", 8: "hit", 9: "hit", 10: "hit", 11: "hit"},
    13: {2: "stand", 3: "stand", 4: "stand", 5: "stand", 6: "stand", 7: "hit", 8: "hit", 9: "hit", 10: "hit", 11: "hit"},
    14: {2: "stand", 3: "stand", 4: "stand", 5: "stand", 6: "stand", 7: "hit", 8: "hit", 9: "hit", 10: "hit", 11: "hit"},
    15: {2: "stand", 3: "stand", 4: "stand", 5: "stand", 6: "stand", 7: "hit", 8: "hit", 9: "hit", 10: "hit", 11: "hit"},
    16: {2: "stand", 3: "stand", 4: "stand", 5: "stand", 6: "stand", 7: "hit", 8: "hit", 9: "hit", 10: "hit", 11: "hit"},
    17: {2: "stand", 3: "stand", 4: "stand", 5: "stand", 6: "stand", 7: "stand", 8: "stand", 9: "stand", 10: "stand", 11: "stand"},
}


PracticeCell = tuple[int, int]


def practice_cells() -> list[PracticeCell]:
    cells = []
    for player_total, dealer_table in HARD_TOTAL_BASIC_STRATEGY.items():
        for dealer_upcard in dealer_table:
            cells.append((player_total, dealer_upcard))
    return cells


def play_training_hand(
    env: BlackjackEnv,
    agent: QLearningAgent,
    epsilon: float,
    practice_cell: PracticeCell | None = None,
) -> float:
    """Play one hand while learning from every action."""

    if practice_cell is None:
        state = env.reset()
    else:
        player_total, dealer_upcard = practice_cell
        state = env.reset_to_hard_total(player_total, dealer_upcard)

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


def train(episodes: int, seed: int, practice_ratio: float) -> tuple[QLearningAgent, list[dict[str, float | int]]]:
    """Train the agent and keep progress logs."""

    env = BlackjackEnv(seed=seed)
    agent = QLearningAgent(seed=seed + 1)
    practice_rng = Random(seed + 2)
    cells = practice_cells()
    logs: list[dict[str, float | int]] = []
    window_profit = 0.0
    log_every = max(1, episodes // 20)

    for episode in range(1, episodes + 1):
        # Explore heavily at first, then settle into the learned policy.
        epsilon = max(0.01, 1.0 - episode / (episodes * 0.5))
        practice_cell = None
        if practice_rng.random() < practice_ratio:
            practice_cell = cells[(episode - 1) % len(cells)]

        window_profit += play_training_hand(env, agent, epsilon, practice_cell)

        if episode % log_every == 0:
            logs.append(
                {
                    "episode": episode,
                    "epsilon": round(epsilon, 4),
                    "practice_ratio": practice_ratio,
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

    print("\nLearned policy, no usable ace:")
    print("player_total | " + " | ".join(f"{label:>5}" for label in DEALER_LABELS))
    for player_total in range(12, 22):
        actions = []
        for dealer_upcard in DEALER_CARDS:
            state = State(player_total, dealer_upcard, False, can_double=True)
            actions.append(agent.best_action(state, ("hit", "stand")))
        print(f"{player_total:>12} | " + " | ".join(f"{action:>5}" for action in actions))


def basic_strategy_accuracy(agent: QLearningAgent) -> dict[str, float | int]:
    """Compare learned hard-total actions with the known basic strategy table."""

    matches = 0
    total = 0

    for player_total, dealer_table in HARD_TOTAL_BASIC_STRATEGY.items():
        for dealer_upcard, expected_action in dealer_table.items():
            state = State(player_total, dealer_upcard, False, can_double=True)
            learned_action = agent.best_action(state, ("hit", "stand"))
            if learned_action == expected_action:
                matches += 1
            total += 1

    return {
        "matches": matches,
        "total": total,
        "accuracy": round(matches / total, 4),
    }


def print_basic_strategy_accuracy(agent: QLearningAgent) -> None:
    score = basic_strategy_accuracy(agent)
    percent = score["accuracy"] * 100
    print(
        "\nHard-total basic strategy match: "
        f"{score['matches']} / {score['total']} cells = {percent:.2f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a basic Blackjack Q-learning player.")
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--eval-hands", type=int, default=10_000)
    parser.add_argument("--out", type=Path, default=Path("results_basic"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--practice-ratio",
        type=float,
        default=0.5,
        help="Fraction of training hands started from hard-total practice states.",
    )
    args = parser.parse_args()

    agent, logs = train(episodes=args.episodes, seed=args.seed, practice_ratio=args.practice_ratio)
    summary = evaluate(agent, hands=args.eval_hands, seed=args.seed + 999)
    strategy_score = basic_strategy_accuracy(agent)
    summary["practice_ratio"] = args.practice_ratio
    summary["hard_total_strategy_matches"] = strategy_score["matches"]
    summary["hard_total_strategy_total"] = strategy_score["total"]
    summary["hard_total_strategy_accuracy"] = strategy_score["accuracy"]

    write_csv(args.out / "training_log.csv", logs)
    write_csv(args.out / "evaluation_summary.csv", [summary])

    print(summary)
    print_policy(agent)
    print_basic_strategy_accuracy(agent)
    print(f"\nWrote logs to: {args.out}")


if __name__ == "__main__":
    main()
