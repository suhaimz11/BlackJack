from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .agent import QLearningAgent
from .env import BlackjackEnv, State


COUNT_VARIANTS = [
    {
        "name": "finite_shoe_flat_bet",
        "use_count_in_state": False,
        "variable_bet": False,
    },
    {
        "name": "count_state_flat_bet",
        "use_count_in_state": True,
        "variable_bet": False,
    },
    {
        "name": "count_state_variable_bet",
        "use_count_in_state": True,
        "variable_bet": True,
    },
]

COUNT_BUCKETS = list(range(-5, 6))


def play_training_hand(env: BlackjackEnv, agent: QLearningAgent, epsilon: float) -> float:
    """Play one counted-shoe hand while learning."""

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


def train(
    episodes: int,
    seed: int,
    decks: int,
    use_count_in_state: bool,
    variable_bet: bool,
    initial_agent: QLearningAgent | None = None,
) -> tuple[QLearningAgent, list[dict[str, float | int]]]:
    """Train one finite-shoe variant."""

    env = BlackjackEnv(
        seed=seed,
        decks=decks,
        use_count_in_state=use_count_in_state,
        variable_bet=variable_bet,
    )
    agent = initial_agent if initial_agent is not None else QLearningAgent(seed=seed + 1)
    logs: list[dict[str, float | int]] = []
    window_profit = 0.0
    log_every = max(1, episodes // 20)

    for episode in range(1, episodes + 1):
        epsilon = max(0.01, 1.0 - episode / (episodes * 0.5))
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


def pretrain_count_agent(episodes: int, seed: int, decks: int) -> QLearningAgent:
    """Learn a no-count policy, then copy it into every count bucket."""

    base_agent, _ = train(
        episodes=episodes,
        seed=seed,
        decks=decks,
        use_count_in_state=False,
        variable_bet=False,
    )
    count_agent = QLearningAgent(seed=seed + 1000)

    for (state, action), value in base_agent.q.items():
        for bucket in COUNT_BUCKETS:
            count_state = State(
                player_total=state.player_total,
                dealer_upcard=state.dealer_upcard,
                usable_ace=state.usable_ace,
                true_count_bucket=bucket,
            )
            count_agent.q[(count_state, action)] = value

    return count_agent


def evaluate(
    agent: QLearningAgent,
    hands: int,
    seed: int,
    decks: int,
    use_count_in_state: bool,
    variable_bet: bool,
) -> dict[str, float | int]:
    """Evaluate one finite-shoe variant greedily."""

    env = BlackjackEnv(
        seed=seed,
        decks=decks,
        use_count_in_state=use_count_in_state,
        variable_bet=variable_bet,
    )
    profit = 0.0
    total_bet = 0
    wins = 0
    losses = 0
    pushes = 0

    for _ in range(hands):
        state = env.reset()
        total_bet += env.bet
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
        "decks": decks,
        "use_count_in_state": use_count_in_state,
        "variable_bet": variable_bet,
        "total_profit": round(profit, 4),
        "avg_profit_per_hand": round(profit / hands, 6),
        "avg_initial_bet": round(total_bet / hands, 4),
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


def print_count_policy_sample(agent: QLearningAgent) -> None:
    """Print a small sample showing that count bucket can change decisions."""

    count_buckets = [-2, 0, 2, 4]
    dealer_cards = [2, 6, 10, 11]
    dealer_labels = ["2", "6", "10", "A"]

    print("\nLearned policy sample, hard 16:")
    print("count | " + " | ".join(f"{label:>5}" for label in dealer_labels))
    for bucket in count_buckets:
        actions = []
        for dealer_upcard in dealer_cards:
            state = State(16, dealer_upcard, False, bucket, can_double=True)
            actions.append(agent.best_action(state, ("hit", "stand", "double")))
        print(f"{bucket:>5} | " + " | ".join(f"{action:>5}" for action in actions))


def run_variant(
    name: str,
    episodes: int,
    eval_hands: int,
    decks: int,
    seed: int,
    use_count_in_state: bool,
    variable_bet: bool,
    pretrain_episodes: int,
) -> tuple[QLearningAgent, list[dict[str, float | int | str]], dict[str, float | int | str | bool]]:
    initial_agent = None
    if use_count_in_state and pretrain_episodes > 0:
        initial_agent = pretrain_count_agent(
            episodes=pretrain_episodes,
            seed=seed + 5000,
            decks=decks,
        )

    agent, logs = train(
        episodes=episodes,
        seed=seed,
        decks=decks,
        use_count_in_state=use_count_in_state,
        variable_bet=variable_bet,
        initial_agent=initial_agent,
    )
    for row in logs:
        row["variant"] = name
        row["pretrain_episodes"] = pretrain_episodes if use_count_in_state else 0

    summary = evaluate(
        agent=agent,
        hands=eval_hands,
        seed=seed + 999,
        decks=decks,
        use_count_in_state=use_count_in_state,
        variable_bet=variable_bet,
    )
    summary["variant"] = name
    summary["pretrain_episodes"] = pretrain_episodes if use_count_in_state else 0
    return agent, logs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Blackjack point-count Q-learning player.")
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--eval-hands", type=int, default=20_000)
    parser.add_argument("--decks", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("results_count"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pretrain-episodes",
        type=int,
        default=0,
        help="No-count episodes used to initialize count-state Q-values.",
    )
    parser.add_argument(
        "--mode",
        choices=["count", "compare"],
        default="count",
        help="count runs the required count+variable-bet scenario; compare runs three diagnostic variants.",
    )
    args = parser.parse_args()

    if args.mode == "compare":
        all_logs: list[dict[str, float | int | str]] = []
        summaries: list[dict[str, float | int | str | bool]] = []
        last_count_agent: QLearningAgent | None = None

        for index, variant in enumerate(COUNT_VARIANTS):
            agent, logs, summary = run_variant(
                name=variant["name"],
                episodes=args.episodes,
                eval_hands=args.eval_hands,
                decks=args.decks,
                seed=args.seed + index * 100,
                use_count_in_state=variant["use_count_in_state"],
                variable_bet=variant["variable_bet"],
                pretrain_episodes=args.pretrain_episodes,
            )
            all_logs.extend(logs)
            summaries.append(summary)
            print(summary)
            if variant["use_count_in_state"]:
                last_count_agent = agent

        write_csv(args.out / "training_log.csv", all_logs)
        write_csv(args.out / "evaluation_summary.csv", summaries)
        if last_count_agent is not None:
            print_count_policy_sample(last_count_agent)
        print(f"\nWrote logs to: {args.out}")
        return

    agent, logs, summary = run_variant(
        name="count_state_variable_bet",
        episodes=args.episodes,
        eval_hands=args.eval_hands,
        decks=args.decks,
        seed=args.seed,
        use_count_in_state=True,
        variable_bet=True,
        pretrain_episodes=args.pretrain_episodes,
    )

    write_csv(args.out / "training_log.csv", logs)
    write_csv(args.out / "evaluation_summary.csv", [summary])

    print(summary)
    print_count_policy_sample(agent)
    print(f"\nWrote logs to: {args.out}")


if __name__ == "__main__":
    main()
