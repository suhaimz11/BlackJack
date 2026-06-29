# Blackjack Reinforcement Learning

This project starts with Task P3.1 from the portfolio exam.

We are building a self-learning Blackjack player step by step.

## Step 1: Basic Strategy Scenario

The first scenario targets the Basic Strategy described in Edward O. Thorp's *Beat the Dealer* (1966). In normal Blackjack, a basic strategy player decides what to do from information like:

- the player's current hand total
- the dealer's visible card
- whether the player has a usable ace

In the current version, the player can choose:

- `hit`
- `stand`
- `double`
- `split`

The strategy is not hard-coded by default. The program uses Q-learning to learn which action is better in each state. Thorp-style Basic Strategy tables are used as reference diagnostics for validation, not as the default final policy. Doubling is available only as the first action of a hand.

To avoid unrealistic double-down choices, doubling is restricted to common double-down situations:

- hard totals 9, 10, and 11
- soft totals 13 through 18

Pair splitting is available when the first two cards have the same Blackjack value. This implementation allows one split per original hand. Insurance is not offered, following the Basic Strategy instruction to never insure.

During training, the program also uses practice states. This means some hands begin from important hard-total situations such as hard 12 vs dealer 4 or hard 16 vs dealer 10. These practice hands help the agent learn the Basic Strategy table more evenly.

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

- `src/blackjack_rl/mc_agent.py`
  - Implements a first-visit Monte Carlo player.
  - Used as a Basic Strategy comparison algorithm.

- `src/blackjack_rl/train_basic.py`
  - Trains the agent.
  - Evaluates the learned policy.
  - Writes result CSV files.

- `src/blackjack_rl/train_basic_mc.py`
  - Trains the Basic Strategy scenario with first-visit Monte Carlo.
  - Uses the same Basic Strategy diagnostics as the Q-learning runner.

- `src/blackjack_rl/train_count.py`
  - Trains the point-count scenario.
  - Uses a finite shoe and Complete Point-Count style card weights.
  - Adjusts the bet based on the true count.

- `src/blackjack_rl/train_variations.py`
  - Evaluates two rule variations:
    - dealer hits soft 17
    - blackjack pays 6:5
  - Tests both Basic Strategy MC and CPCS Q-learning under each variation.

- `src/blackjack_rl/train_improved_cpcs.py`
  - Compares standard 6-deck CPCS with an improved single-deck CPCS setup.

## Run

From this folder:

```powershell
python -m src.blackjack_rl.train_basic --episodes 50000 --eval-hands 10000 --out results_basic
```

Practice states are enabled by default:

```powershell
python -m src.blackjack_rl.train_basic --episodes 50000 --eval-hands 10000 --practice-ratio 0.5 --out results_basic
```

Use `--practice-ratio 0` to train only from normal random hands.

To optionally initialize selected double-down Q-values from the reference table:

```powershell
python -m src.blackjack_rl.train_basic --episodes 50000 --eval-hands 10000 --practice-ratio 0.5 --use-basic-priors --out results_basic_priors
```

To compare Basic Strategy with first-visit Monte Carlo:

```powershell
python -m src.blackjack_rl.train_basic_mc --episodes 100000 --eval-hands 20000 --practice-ratio 0.8 --out results_basic_mc
```

The Monte Carlo runner samples practice states by category:

- 40% hard/soft draw-stand states
- 30% double-down states
- 30% pair-splitting states

## Step 2: Complete Point-Count Scenario

The second scenario adds a finite Blackjack shoe and a Complete Point-Count style system.

The implemented point-count card weights are:

- cards 2-6 add `+1`
- cards 7-9 add `0`
- cards 10 and ace add `-1`

Following Thorp's CPCS description, the program converts the running count into a high-low index:

```text
high-low index = 100 * running count / unseen cards
```

The state includes a rounded high-low index bucket.

For the portfolio task, this is the implemented Complete Point-Count style system:

- finite 6-deck shoe by default
- reshuffle after 75% shoe penetration
- visible cards update the running count
- the dealer hole card is counted only when revealed
- high-low index is rounded and clipped into buckets from `-20` to `20`
- the RL state becomes player total, dealer upcard, usable ace, high-low index bucket, whether double is legal, and whether split is legal

The bet also changes:

- index 2 or lower: bet 1 unit
- index 4 or 5: bet 2 units
- index 6 or 7: bet 3 units
- index 8 or 9: bet 4 units
- index 10 or higher: bet 5 units

Run the point-count scenario:

```powershell
python -m src.blackjack_rl.train_count --episodes 100000 --eval-hands 20000 --out results_count
```

To compare the effect of count state and variable betting separately:

```powershell
python -m src.blackjack_rl.train_count --mode compare --episodes 100000 --eval-hands 20000 --out results_count_compare
```

To initialize count-state agents from a learned no-count policy:

```powershell
python -m src.blackjack_rl.train_count --mode compare --pretrain-episodes 100000 --episodes 100000 --eval-hands 20000 --out results_count_compare_pretrained
```

For a quick point-count smoke test:

```powershell
python -m src.blackjack_rl.train_count --episodes 5000 --eval-hands 1000 --out results_count_smoke
```

## Step 3: Rule Variations

The third scenario examines two rule changes:

- dealer hits soft 17
- blackjack pays 6:5

Run both rule variations:

```powershell
python -m src.blackjack_rl.train_variations --episodes-basic 200000 --episodes-cpcs 200000 --eval-hands 30000 --out results_variations
```

For a quick smoke test:

```powershell
python -m src.blackjack_rl.train_variations --episodes-basic 1000 --episodes-cpcs 1000 --eval-hands 500 --out results_variations_smoke
```

## Step 4: Improved CPCS

The fourth scenario improves the CPCS system by using a single-deck shoe. Card counting is more informative with fewer decks because each visible card changes the remaining shoe composition more strongly.

Run the improved CPCS comparison:

```powershell
python -m src.blackjack_rl.train_improved_cpcs --pretrain-episodes 300000 --episodes 300000 --eval-hands 30000 --out results_improved_cpcs
```

For a quick smoke test:

```powershell
python -m src.blackjack_rl.train_improved_cpcs --pretrain-episodes 1000 --episodes 1000 --eval-hands 500 --out results_improved_cpcs_smoke
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
- `bet_by_index_band.csv` for point-count runs

These files will later support the paper in Task P3.2.
