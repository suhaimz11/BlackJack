from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .train_count import run_variant, write_csv, print_bet_by_count, print_count_policy_tables
from .env import BlackjackEnv


IMPROVED_CPCS_VARIANTS = [
    {
        "name": "standard_cpcs_6_decks",
        "decks": 6,
        "skip_unfavorable": False,
    },
    {
        "name": "improved_cpcs_backcounting",
        "decks": 6,
        "skip_unfavorable": True,
    },
]


def evaluate_with_backcounting(
    agent,
    hands: int,
    seed: int,
    decks: int,
    skip_index_at_or_below: float = 2.0,
) -> tuple[dict[str, float | int | str | bool], list[dict[str, float | int | str]]]:
    """Evaluate CPCS while skipping hands when the index is unfavorable."""

    env = BlackjackEnv(seed=seed, decks=decks, use_count_in_state=True, variable_bet=True)
    profit = 0.0
    total_bet = 0
    wins = 0
    losses = 0
    pushes = 0
    skipped = 0
    bets_by_index: dict[str, list[int]] = defaultdict(list)

    for _ in range(hands):
        current_index = env.current_high_low_index()
        if current_index <= skip_index_at_or_below:
            skipped += 1
            bets_by_index["skipped_<=2"].append(0)
            env.shoe.maybe_shuffle()
            # Burn one visible card to advance the shoe/count without risking money.
            env.draw(visible=True)
            pushes += 1
            continue

        state = env.reset()
        total_bet += env.bet
        bets_by_index[env.bet_index_band].append(env.bet)
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

    played_hands = hands - skipped
    summary = {
        "hands": hands,
        "played_hands": played_hands,
        "skipped_hands": skipped,
        "decks": decks,
        "use_count_in_state": True,
        "variable_bet": True,
        "skip_unfavorable": True,
        "total_profit": round(profit, 4),
        "avg_profit_per_hand": round(profit / hands, 6),
        "avg_profit_per_played_hand": round(profit / played_hands, 6) if played_hands else 0,
        "avg_initial_bet": round(total_bet / played_hands, 4) if played_hands else 0,
        "win_rate": round(wins / hands, 6),
        "loss_rate": round(losses / hands, 6),
        "push_rate": round(pushes / hands, 6),
        "learned_state_actions": len(agent.q),
    }
    bet_rows = [
        {
            "betting_index_band": bucket,
            "avg_bet": round(mean(bets), 4),
            "hands": len(bets),
        }
        for bucket, bets in sorted(bets_by_index.items())
    ]
    return summary, bet_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare standard and improved CPCS systems.")
    parser.add_argument("--pretrain-episodes", type=int, default=300_000)
    parser.add_argument("--episodes", type=int, default=300_000)
    parser.add_argument("--eval-hands", type=int, default=30_000)
    parser.add_argument("--out", type=Path, default=Path("results_improved_cpcs"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_logs: list[dict] = []
    summaries: list[dict] = []
    all_bet_rows: list[dict] = []
    improved_agent = None

    for index, variant in enumerate(IMPROVED_CPCS_VARIANTS):
        agent, logs, summary, bet_rows = run_variant(
            name=variant["name"],
            episodes=args.episodes,
            eval_hands=args.eval_hands,
            decks=variant["decks"],
            seed=args.seed + index * 100,
            use_count_in_state=True,
            variable_bet=True,
            pretrain_episodes=args.pretrain_episodes,
        )

        if variant["skip_unfavorable"]:
            summary, bet_rows = evaluate_with_backcounting(
                agent=agent,
                hands=args.eval_hands,
                seed=args.seed + index * 100 + 999,
                decks=variant["decks"],
            )
            summary["variant"] = variant["name"]
            summary["pretrain_episodes"] = args.pretrain_episodes
            for row in bet_rows:
                row["variant"] = variant["name"]

        for row in logs:
            row["decks"] = variant["decks"]
        for row in bet_rows:
            row["decks"] = variant["decks"]
        summary["decks"] = variant["decks"]
        summary["skip_unfavorable"] = variant["skip_unfavorable"]

        all_logs.extend(logs)
        summaries.append(summary)
        all_bet_rows.extend(bet_rows)
        print(summary)

        if variant["skip_unfavorable"]:
            improved_agent = agent

    write_csv(args.out / "training_log.csv", all_logs)
    write_csv(args.out / "evaluation_summary.csv", summaries)
    write_csv(args.out / "bet_by_index_band.csv", all_bet_rows)

    if improved_agent is not None:
        print_count_policy_tables(improved_agent)
    print_bet_by_count(all_bet_rows)
    print(f"\nWrote logs to: {args.out}")


if __name__ == "__main__":
    main()
