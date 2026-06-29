from __future__ import annotations

import argparse
import csv
from pathlib import Path
from random import Random

from .agent import QLearningAgent
from .env import BlackjackEnv, State, can_double_hand, can_split_hand


DEALER_CARDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
DEALER_LABELS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"]

# Standard hard-total Basic Strategy, following the Thorp-style reference table.
# A dealer ace is represented as 11 in the code.
HARD_TOTAL_BASIC_STRATEGY: dict[int, dict[int, str]] = {
    12: {2: "hit", 3: "hit", 4: "stand", 5: "stand", 6: "stand", 7: "hit", 8: "hit", 9: "hit", 10: "hit", 11: "hit"},
    13: {2: "stand", 3: "stand", 4: "stand", 5: "stand", 6: "stand", 7: "hit", 8: "hit", 9: "hit", 10: "hit", 11: "hit"},
    14: {2: "stand", 3: "stand", 4: "stand", 5: "stand", 6: "stand", 7: "hit", 8: "hit", 9: "hit", 10: "hit", 11: "hit"},
    15: {2: "stand", 3: "stand", 4: "stand", 5: "stand", 6: "stand", 7: "hit", 8: "hit", 9: "hit", 10: "hit", 11: "hit"},
    16: {2: "stand", 3: "stand", 4: "stand", 5: "stand", 6: "stand", 7: "hit", 8: "hit", 9: "hit", 10: "hit", 11: "hit"},
    17: {2: "stand", 3: "stand", 4: "stand", 5: "stand", 6: "stand", 7: "stand", 8: "stand", 9: "stand", 10: "stand", 11: "stand"},
}

# Simplified pair-splitting rule described in Thorp's Basic Strategy chapter:
# always split aces/eights; never split fours/fives/tens; split other pairs
# against dealer 3 through 7.
PAIR_SPLIT_STRATEGY: dict[int, dict[int, bool]] = {
    pair: {
        dealer: (
            pair in (8, 11)
            or (pair not in (4, 5, 10) and 3 <= dealer <= 7)
        )
        for dealer in DEALER_CARDS
    }
    for pair in DEALER_CARDS
}

SOFT_TOTAL_STAND_STRATEGY: dict[int, dict[int, str]] = {
    soft_total: {
        dealer: (
            "stand"
            if soft_total >= 19 or (soft_total == 18 and dealer <= 8)
            else "hit"
        )
        for dealer in DEALER_CARDS
    }
    for soft_total in range(13, 22)
}

DOUBLE_DOWN_STRATEGY: dict[str, dict[int, dict[int, bool]]] = {
    "hard": {
        9: {dealer: 3 <= dealer <= 6 for dealer in DEALER_CARDS},
        10: {dealer: 2 <= dealer <= 9 for dealer in DEALER_CARDS},
        11: {dealer: True for dealer in DEALER_CARDS},
    },
    "soft": {
        13: {dealer: dealer in (5, 6) for dealer in DEALER_CARDS},
        14: {dealer: dealer in (5, 6) for dealer in DEALER_CARDS},
        15: {dealer: 4 <= dealer <= 6 for dealer in DEALER_CARDS},
        16: {dealer: 4 <= dealer <= 6 for dealer in DEALER_CARDS},
        17: {dealer: 3 <= dealer <= 6 for dealer in DEALER_CARDS},
        18: {dealer: 2 <= dealer <= 6 for dealer in DEALER_CARDS},
    },
}


PracticeCell = tuple[str, int, int]


def practice_cells_by_category() -> dict[str, list[PracticeCell]]:
    categories: dict[str, list[PracticeCell]] = {
        "draw_stand": [],
        "double": [],
        "split": [],
    }

    for player_total, dealer_table in HARD_TOTAL_BASIC_STRATEGY.items():
        for dealer_upcard in dealer_table:
            categories["draw_stand"].append(("hard", player_total, dealer_upcard))
    for soft_total, dealer_table in SOFT_TOTAL_STAND_STRATEGY.items():
        for dealer_upcard in dealer_table:
            categories["draw_stand"].append(("soft", soft_total, dealer_upcard))
    for hard_total, dealer_table in DOUBLE_DOWN_STRATEGY["hard"].items():
        for dealer_upcard in dealer_table:
            categories["double"].append(("double_hard", hard_total, dealer_upcard))
    for soft_total, dealer_table in DOUBLE_DOWN_STRATEGY["soft"].items():
        for dealer_upcard in dealer_table:
            categories["double"].append(("double_soft", soft_total, dealer_upcard))
    for pair_card, dealer_table in PAIR_SPLIT_STRATEGY.items():
        for dealer_upcard in dealer_table:
            categories["split"].append(("pair", pair_card, dealer_upcard))

    return categories


def practice_cells() -> list[PracticeCell]:
    cells = []
    categories = practice_cells_by_category()
    cells.extend(categories["draw_stand"])
    cells.extend(categories["double"] * 3)
    cells.extend(categories["split"])
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
        kind, value, dealer_upcard = practice_cell
        if kind == "pair":
            state = env.reset_to_pair(value, dealer_upcard)
        elif kind in ("soft", "double_soft"):
            state = env.reset_to_soft_total(value, dealer_upcard)
        else:
            state = env.reset_to_hard_total(value, dealer_upcard)

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


def initialize_basic_strategy_priors(agent: QLearningAgent, value: float = 0.2) -> None:
    """Give the learner a small Table 3.3 prior for double-down decisions."""

    for hard_total, dealer_table in DOUBLE_DOWN_STRATEGY["hard"].items():
        hand = [2, 7] if hard_total == 9 else ([4, 6] if hard_total == 10 else [5, 6])
        for dealer_upcard, should_double in dealer_table.items():
            state = State(
                hard_total,
                dealer_upcard,
                False,
                can_double=can_double_hand(hand),
                can_split=False,
            )
            agent.q[(state, "double")] = value if should_double else -value

    for soft_total, dealer_table in DOUBLE_DOWN_STRATEGY["soft"].items():
        hand = [11, soft_total - 11]
        for dealer_upcard, should_double in dealer_table.items():
            state = State(
                soft_total,
                dealer_upcard,
                True,
                can_double=can_double_hand(hand),
                can_split=False,
            )
            agent.q[(state, "double")] = value if should_double else -value


def train(
    episodes: int,
    seed: int,
    practice_ratio: float,
    use_basic_priors: bool,
) -> tuple[QLearningAgent, list[dict[str, float | int]]]:
    """Train the agent and keep progress logs."""

    env = BlackjackEnv(seed=seed)
    agent = QLearningAgent(seed=seed + 1)
    if use_basic_priors:
        initialize_basic_strategy_priors(agent)
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
                    "use_basic_priors": use_basic_priors,
                    "practice_cells": len(cells),
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
            state = State(
                player_total,
                dealer_upcard,
                False,
                can_double=can_double_hand([10, player_total - 10]),
                can_split=False,
            )
            actions.append(agent.best_action(state, ("hit", "stand")))
        print(f"{player_total:>12} | " + " | ".join(f"{action:>5}" for action in actions))


def print_double_policy(agent: QLearningAgent) -> None:
    """Print learned double-down choices for hard and soft double states."""

    print("\nLearned hard double-down policy:")
    print("player_total | " + " | ".join(f"{label:>5}" for label in DEALER_LABELS))
    for hard_total in sorted(DOUBLE_DOWN_STRATEGY["hard"]):
        hand = [2, 7] if hard_total == 9 else ([4, 6] if hard_total == 10 else [5, 6])
        actions = []
        for dealer_upcard in DEALER_CARDS:
            state = State(
                hard_total,
                dealer_upcard,
                False,
                can_double=can_double_hand(hand),
                can_split=False,
            )
            best = agent.best_action(state, ("hit", "stand", "double"))
            actions.append("D" if best == "double" else "-")
        print(f"{hard_total:>12} | " + " | ".join(f"{action:>5}" for action in actions))

    print("\nLearned soft double-down policy:")
    print("soft_total   | " + " | ".join(f"{label:>5}" for label in DEALER_LABELS))
    for soft_total in sorted(DOUBLE_DOWN_STRATEGY["soft"]):
        hand = [11, soft_total - 11]
        actions = []
        for dealer_upcard in DEALER_CARDS:
            state = State(
                soft_total,
                dealer_upcard,
                True,
                can_double=can_double_hand(hand),
                can_split=False,
            )
            best = agent.best_action(state, ("hit", "stand", "double"))
            actions.append("D" if best == "double" else "-")
        print(f"{soft_total:>12} | " + " | ".join(f"{action:>5}" for action in actions))


def basic_strategy_accuracy(agent: QLearningAgent) -> dict[str, float | int]:
    """Compare learned hard-total actions with the known basic strategy table."""

    matches = 0
    total = 0

    for player_total, dealer_table in HARD_TOTAL_BASIC_STRATEGY.items():
        for dealer_upcard, expected_action in dealer_table.items():
            state = State(player_total, dealer_upcard, False, can_double=can_double_hand([10, player_total - 10]))
            learned_action = agent.best_action(state, ("hit", "stand"))
            if learned_action == expected_action:
                matches += 1
            total += 1

    return {
        "matches": matches,
        "total": total,
        "accuracy": round(matches / total, 4),
    }


def pair_split_accuracy(agent: QLearningAgent) -> dict[str, float | int]:
    """Compare learned split/no-split choices with the pair-splitting rule."""

    matches = 0
    total = 0

    for pair_card, dealer_table in PAIR_SPLIT_STRATEGY.items():
        for dealer_upcard, should_split in dealer_table.items():
            hand = [pair_card, pair_card]
            total_value, usable_ace = (12, True) if pair_card == 11 else (pair_card * 2, False)
            state = State(
                total_value,
                dealer_upcard,
                usable_ace,
                can_double=can_double_hand(hand),
                can_split=can_split_hand(hand),
            )
            legal_actions = ["hit", "stand"]
            if can_double_hand(hand):
                legal_actions.append("double")
            legal_actions.append("split")
            learned_action = agent.best_action(state, tuple(legal_actions))
            learned_split = learned_action == "split"
            if learned_split == should_split:
                matches += 1
            total += 1

    return {
        "matches": matches,
        "total": total,
        "accuracy": round(matches / total, 4),
    }


def soft_strategy_accuracy(agent: QLearningAgent) -> dict[str, float | int]:
    """Compare learned soft-total hit/stand choices with the reference rule."""

    matches = 0
    total = 0

    for soft_total, dealer_table in SOFT_TOTAL_STAND_STRATEGY.items():
        for dealer_upcard, expected_action in dealer_table.items():
            state = State(
                soft_total,
                dealer_upcard,
                True,
                can_double=can_double_hand([11, soft_total - 11]),
                can_split=False,
            )
            learned_action = agent.best_action(state, ("hit", "stand"))
            if learned_action == expected_action:
                matches += 1
            total += 1

    return {
        "matches": matches,
        "total": total,
        "accuracy": round(matches / total, 4),
    }


def double_strategy_accuracy(agent: QLearningAgent) -> dict[str, float | int]:
    """Compare learned double/no-double choices with the double-down rule."""

    matches = 0
    total = 0

    for hard_total, dealer_table in DOUBLE_DOWN_STRATEGY["hard"].items():
        hand = [4, 6] if hard_total == 10 else [5, 6]
        for dealer_upcard, should_double in dealer_table.items():
            state = State(
                hard_total,
                dealer_upcard,
                False,
                can_double=can_double_hand(hand),
                can_split=False,
            )
            learned_action = agent.best_action(state, ("hit", "stand", "double"))
            if (learned_action == "double") == should_double:
                matches += 1
            total += 1

    for soft_total, dealer_table in DOUBLE_DOWN_STRATEGY["soft"].items():
        hand = [11, soft_total - 11]
        for dealer_upcard, should_double in dealer_table.items():
            state = State(
                soft_total,
                dealer_upcard,
                True,
                can_double=can_double_hand(hand),
                can_split=False,
            )
            learned_action = agent.best_action(state, ("hit", "stand", "double"))
            if (learned_action == "double") == should_double:
                matches += 1
            total += 1

    return {
        "matches": matches,
        "total": total,
        "accuracy": round(matches / total, 4),
    }


def print_basic_strategy_accuracy(agent: QLearningAgent) -> None:
    hard_score = basic_strategy_accuracy(agent)
    soft_score = soft_strategy_accuracy(agent)
    double_score = double_strategy_accuracy(agent)
    split_score = pair_split_accuracy(agent)
    hard_percent = hard_score["accuracy"] * 100
    soft_percent = soft_score["accuracy"] * 100
    double_percent = double_score["accuracy"] * 100
    split_percent = split_score["accuracy"] * 100
    print(
        "\nHard-total basic strategy match: "
        f"{hard_score['matches']} / {hard_score['total']} cells = {hard_percent:.2f}%"
    )
    print(
        "Soft-total basic strategy match: "
        f"{soft_score['matches']} / {soft_score['total']} cells = {soft_percent:.2f}%"
    )
    print(
        "Double-down strategy match: "
        f"{double_score['matches']} / {double_score['total']} cells = {double_percent:.2f}%"
    )
    print(
        "Pair-splitting strategy match: "
        f"{split_score['matches']} / {split_score['total']} cells = {split_percent:.2f}%"
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
    parser.add_argument(
        "--use-basic-priors",
        action="store_true",
        help="Initialize selected Q-values from Thorp-style double-down rules.",
    )
    args = parser.parse_args()

    agent, logs = train(
        episodes=args.episodes,
        seed=args.seed,
        practice_ratio=args.practice_ratio,
        use_basic_priors=args.use_basic_priors,
    )
    summary = evaluate(agent, hands=args.eval_hands, seed=args.seed + 999)
    strategy_score = basic_strategy_accuracy(agent)
    soft_score = soft_strategy_accuracy(agent)
    double_score = double_strategy_accuracy(agent)
    split_score = pair_split_accuracy(agent)
    summary["practice_ratio"] = args.practice_ratio
    summary["use_basic_priors"] = args.use_basic_priors
    summary["hard_total_strategy_matches"] = strategy_score["matches"]
    summary["hard_total_strategy_total"] = strategy_score["total"]
    summary["hard_total_strategy_accuracy"] = strategy_score["accuracy"]
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
