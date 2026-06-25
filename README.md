# Blackjack Reinforcement Learning

This project starts with Task P3.1 from the portfolio exam.

We are building a self-learning Blackjack player step by step.

## Step 1: Basic Strategy Scenario

In normal Blackjack, a basic strategy player decides what to do from information like:

- the player's current hand total
- the dealer's visible card
- whether the player has a usable ace

In this first version, the player can choose only:

- `hit`
- `stand`

The strategy is not hard-coded. Instead, the program uses Q-learning to learn which action is better in each state.

## Files

- `src/blackjack_rl/env.py`
  - Implements the Blackjack game.
  - Deals cards.
  - Calculates hand values.
  - Applies `hit` and `stand`.
  - Gives rewards: `+1` for win, `-1` for loss, `0` for push.

- `src/blackjack_rl/agent.py`
  - Implements the Q-learning player.
  - Stores learned values in a Q-table.
  - Chooses actions during training and evaluation.

- `src/blackjack_rl/train_basic.py`
  - Trains the agent.
  - Evaluates the learned policy.
  - Writes result CSV files.

## Run

From this folder:

```powershell
python -m src.blackjack_rl.train_basic --episodes 50000 --eval-hands 10000 --out results_basic
```

For a quicker test:

```powershell
python -m src.blackjack_rl.train_basic --episodes 5000 --eval-hands 1000 --out results_basic_smoke
```

If `python` is not available, use the bundled Codex Python:

```powershell
& 'C:\Users\suhai\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m src.blackjack_rl.train_basic --episodes 5000 --eval-hands 1000 --out results_basic_smoke
```

## Output

The program creates:

- `training_log.csv`
- `evaluation_summary.csv`

These files will later support the paper in Task P3.2.
