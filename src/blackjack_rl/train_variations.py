from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .agent import QLearningAgent
from .env import BlackjackEnv
from .mc_agent import MonteCarloAgent
from .train_basic_mc import play_mc_hand
from .train_basic import practice_cells_by_category
from .train_count import play_training_hand as play_q_hand


VARIATIONS = [
    {
        "name": "dealer_hits_soft_17",
        "dealer_hits_soft_17": True,
        "blackjack_payout": 1.5,
    },
    {
        "name": "blackjack_pays_6_to_5",
        "dealer_hits_soft_17": False,
        "blackjack_payout": 1.2,
    },
]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_basic_mc_variation(
    episodes: int,
    seed: int,
    dealer_hits_soft_17: bool,
    blackjack_payout: float,
) -> tuple[MonteCarloAgent, list[dict]]:
    env = BlackjackEnv(
        seed=seed,
        dealer_hits_soft_17=dealer_hits_soft_17,
        blackjack_payout=blackjack_payout,
    )
    agent = MonteCarloAgent(seed=seed + 1)
    categories = practice_cells_by_category()
    all_cells = categories["draw_stand"] + categories["double"] + categories["split"]
    logs: list[dict] = []
    window_profit = 0.0
    log_every = max(1, episodes // 20)

    for episode in range(1, episodes + 1):
        epsilon = max(0.01, 0.30 * (1 - episode / episodes))
        practice_cell = all_cells[(episode - 1) % len(all_cells)]
        window_profit += play_mc_hand(env, agent, epsilon, practice_cell)

        if episode % log_every == 0:
            logs.append(
                {
                    "algorithm": "first_visit_monte_carlo",
                    "episode": episode,
                    "epsilon": round(epsilon, 4),
                    "avg_profit_last_window": round(window_profit / log_every, 5),
                    "known_state_actions": len(agent.q),
                }
            )
            window_profit = 0.0

    return agent, logs


def train_cpcs_variation(
    episodes: int,
    seed: int,
    dealer_hits_soft_17: bool,
    blackjack_payout: float,
) -> tuple[QLearningAgent, list[dict]]:
    env = BlackjackEnv(
        seed=seed,
        decks=6,
        use_count_in_state=True,
        variable_bet=True,
        dealer_hits_soft_17=dealer_hits_soft_17,
        blackjack_payout=blackjack_payout,
    )
    agent = QLearningAgent(seed=seed + 1)
    logs: list[dict] = []
    window_profit = 0.0
    log_every = max(1, episodes // 20)

    for episode in range(1, episodes + 1):
        epsilon = max(0.01, 1.0 - episode / (episodes * 0.5))
        window_profit += play_q_hand(env, agent, epsilon)

        if episode % log_every == 0:
            logs.append(
                {
                    "algorithm": "q_learning_cpcs",
                    "episode": episode,
                    "epsilon": round(epsilon, 4),
                    "avg_profit_last_window": round(window_profit / log_every, 5),
                    "known_state_actions": len(agent.q),
                }
            )
            window_profit = 0.0

    return agent, logs


def evaluate_agent(agent, env: BlackjackEnv, hands: int) -> dict[str, float | int]:
    profit = 0.0
    total_bet = 0
    wins = losses = pushes = 0

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
        "total_profit": round(profit, 4),
        "avg_profit_per_hand": round(profit / hands, 6),
        "avg_initial_bet": round(total_bet / hands, 4),
        "win_rate": round(wins / hands, 6),
        "loss_rate": round(losses / hands, 6),
        "push_rate": round(pushes / hands, 6),
        "learned_state_actions": len(agent.q),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Blackjack rule variations.")
    parser.add_argument("--episodes-basic", type=int, default=200_000)
    parser.add_argument("--episodes-cpcs", type=int, default=200_000)
    parser.add_argument("--eval-hands", type=int, default=30_000)
    parser.add_argument("--out", type=Path, default=Path("results_variations"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_logs: list[dict] = []
    summaries: list[dict] = []

    for index, variation in enumerate(VARIATIONS):
        base_seed = args.seed + index * 1000

        basic_agent, basic_logs = train_basic_mc_variation(
            episodes=args.episodes_basic,
            seed=base_seed,
            dealer_hits_soft_17=variation["dealer_hits_soft_17"],
            blackjack_payout=variation["blackjack_payout"],
        )
        basic_env = BlackjackEnv(
            seed=base_seed + 500,
            dealer_hits_soft_17=variation["dealer_hits_soft_17"],
            blackjack_payout=variation["blackjack_payout"],
        )
        basic_summary = evaluate_agent(basic_agent, basic_env, args.eval_hands)
        basic_summary["variation"] = variation["name"]
        basic_summary["strategy"] = "basic_strategy_mc"
        summaries.append(basic_summary)
        for row in basic_logs:
            row["variation"] = variation["name"]
            row["strategy"] = "basic_strategy_mc"
        all_logs.extend(basic_logs)
        print(basic_summary)

        cpcs_agent, cpcs_logs = train_cpcs_variation(
            episodes=args.episodes_cpcs,
            seed=base_seed + 100,
            dealer_hits_soft_17=variation["dealer_hits_soft_17"],
            blackjack_payout=variation["blackjack_payout"],
        )
        cpcs_env = BlackjackEnv(
            seed=base_seed + 600,
            decks=6,
            use_count_in_state=True,
            variable_bet=True,
            dealer_hits_soft_17=variation["dealer_hits_soft_17"],
            blackjack_payout=variation["blackjack_payout"],
        )
        cpcs_summary = evaluate_agent(cpcs_agent, cpcs_env, args.eval_hands)
        cpcs_summary["variation"] = variation["name"]
        cpcs_summary["strategy"] = "cpcs_q_learning"
        summaries.append(cpcs_summary)
        for row in cpcs_logs:
            row["variation"] = variation["name"]
            row["strategy"] = "cpcs_q_learning"
        all_logs.extend(cpcs_logs)
        print(cpcs_summary)

    write_csv(args.out / "training_log.csv", all_logs)
    write_csv(args.out / "evaluation_summary.csv", summaries)
    print(f"\nWrote logs to: {args.out}")


if __name__ == "__main__":
    main()
