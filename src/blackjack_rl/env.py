from __future__ import annotations

from dataclasses import dataclass
from random import Random


# In this simple project, a card is stored as its Blackjack value.
# Example: Jack, Queen, King are all stored as 10. Ace starts as 11.
Card = int


@dataclass(frozen=True)
class State:
    """The information our learning player can see before choosing an action."""

    player_total: int
    dealer_upcard: int
    usable_ace: bool
    true_count_bucket: int = 0
    can_double: bool = False


class Shoe:
    """A finite Blackjack shoe that keeps a Hi-Lo running count."""

    def __init__(self, decks: int, penetration: float, rng: Random):
        self.decks = decks
        self.penetration = penetration
        self.rng = rng
        self.cards: list[Card] = []
        self.running_count = 0
        self.shuffle()

    def shuffle(self) -> None:
        cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
        self.cards = cards * 4 * self.decks
        self.rng.shuffle(self.cards)
        self.running_count = 0

    def maybe_shuffle(self) -> None:
        cards_total = 52 * self.decks
        used_fraction = 1 - len(self.cards) / cards_total
        if used_fraction >= self.penetration:
            self.shuffle()

    def draw(self, update_count: bool = True) -> Card:
        if not self.cards:
            self.shuffle()
        card = self.cards.pop()
        if update_count:
            self.running_count += hi_lo_value(card)
        return card

    def count_seen_card(self, card: Card) -> None:
        self.running_count += hi_lo_value(card)

    @property
    def true_count(self) -> float:
        decks_remaining = max(len(self.cards) / 52, 0.25)
        return self.running_count / decks_remaining


def draw_card(rng: Random) -> Card:
    """Draw one random card from an infinite deck."""

    cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
    return rng.choice(cards)


def hi_lo_value(card: Card) -> int:
    """Return the Hi-Lo count value of a card."""

    if 2 <= card <= 6:
        return 1
    if card in (10, 11):
        return -1
    return 0


def count_bucket(true_count: float) -> int:
    """Keep the count state small by rounding and clipping true count."""

    return max(-5, min(5, round(true_count)))


def bet_units(true_count: float, variable_bet: bool) -> int:
    """Conservative bet spread: bet more only when the count is clearly favorable."""

    if not variable_bet:
        return 1
    if true_count < 2:
        return 1
    if true_count < 3:
        return 2
    if true_count < 4:
        return 3
    return 4


def hand_value(hand: list[Card]) -> tuple[int, bool]:
    """Return the best hand total and whether an ace is still counted as 11."""

    total = sum(hand)
    aces = hand.count(11)

    # If the hand busts, convert aces from 11 to 1 until it no longer busts.
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1

    usable_ace = aces > 0
    return total, usable_ace


def is_blackjack(hand: list[Card]) -> bool:
    total, _ = hand_value(hand)
    return len(hand) == 2 and total == 21


def can_double_hand(hand: list[Card]) -> bool:
    """Allow double only on common basic-strategy double-down totals."""

    if len(hand) != 2:
        return False

    total, usable_ace = hand_value(hand)
    if usable_ace:
        return 13 <= total <= 18
    return 9 <= total <= 11


class BlackjackEnv:
    """A minimal Blackjack environment for learning basic strategy."""

    def __init__(
        self,
        seed: int = 7,
        decks: int | None = None,
        penetration: float = 0.75,
        use_count_in_state: bool = False,
        variable_bet: bool = False,
    ):
        self.rng = Random(seed)
        self.shoe = None if decks is None else Shoe(decks, penetration, self.rng)
        self.use_count_in_state = use_count_in_state
        self.variable_bet = variable_bet
        self.player: list[Card] = []
        self.dealer: list[Card] = []
        self.dealer_hole_counted = False
        self.bet = 1
        self.done = False

    def draw(self, visible: bool = True) -> Card:
        if self.shoe is None:
            return draw_card(self.rng)
        return self.shoe.draw(update_count=visible)

    def reveal_dealer_hole(self) -> None:
        if self.shoe is not None and not self.dealer_hole_counted:
            self.shoe.count_seen_card(self.dealer[1])
        self.dealer_hole_counted = True

    def current_true_count(self) -> float:
        if self.shoe is None:
            return 0.0
        return self.shoe.true_count

    def reset(self) -> State:
        """Start a new hand and return the first state."""

        if self.shoe is not None:
            self.shoe.maybe_shuffle()

        self.bet = bet_units(self.current_true_count(), self.variable_bet)
        self.player = [self.draw(), self.draw()]
        self.dealer = [self.draw(), self.draw(visible=False)]
        self.dealer_hole_counted = False
        self.done = False
        return self.state()

    def reset_to_hard_total(self, player_total: int, dealer_upcard: int) -> State:
        """Start a practice hand from a chosen hard total and dealer upcard."""

        if not 12 <= player_total <= 21:
            raise ValueError("Practice player total must be between 12 and 21.")
        if dealer_upcard not in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            raise ValueError("Dealer upcard must be 2-10 or 11 for ace.")

        # Use only non-ace cards so this is definitely a hard total.
        if player_total <= 20:
            self.player = [10, player_total - 10]
        else:
            self.player = [10, 9, 2]

        self.bet = 1
        self.dealer = [dealer_upcard, self.draw(visible=False)]
        self.dealer_hole_counted = False
        self.done = False
        return self.state()

    def state(self) -> State:
        total, usable_ace = hand_value(self.player)
        return State(
            player_total=total,
            dealer_upcard=self.dealer[0],
            usable_ace=usable_ace,
            true_count_bucket=count_bucket(self.current_true_count()) if self.use_count_in_state else 0,
            can_double=can_double_hand(self.player),
        )

    def legal_actions(self) -> tuple[str, ...]:
        if can_double_hand(self.player):
            return ("hit", "stand", "double")
        return ("hit", "stand")

    def step(self, action: str) -> tuple[State | None, float, bool]:
        """Apply one action.

        Returns:
            next_state: the next visible state, or None if the hand is over
            reward: +1 win, -1 loss, 0 push/non-terminal
            done: True if the hand is finished
        """

        if self.done:
            raise RuntimeError("Cannot act after the hand is finished.")

        if action == "hit":
            self.player.append(self.draw())
            player_total, _ = hand_value(self.player)

            if player_total > 21:
                self.done = True
                return None, -float(self.bet), True

            return self.state(), 0.0, False

        if action == "double":
            if not can_double_hand(self.player):
                raise ValueError("Double is not legal for this hand.")
            self.bet *= 2
            self.player.append(self.draw())
            player_total, _ = hand_value(self.player)

            if player_total > 21:
                self.done = True
                return None, -float(self.bet), True

            return self.finish_round()

        if action == "stand":
            return self.finish_round()

        raise ValueError(f"Unknown action: {action}")

    def finish_round(self) -> tuple[None, float, bool]:
        """Let the dealer play, then decide win/loss/push."""

        self.done = True

        if is_blackjack(self.player) and not is_blackjack(self.dealer):
            self.reveal_dealer_hole()
            return None, float(self.bet), True
        if is_blackjack(self.dealer) and not is_blackjack(self.player):
            self.reveal_dealer_hole()
            return None, -float(self.bet), True
        if is_blackjack(self.player) and is_blackjack(self.dealer):
            self.reveal_dealer_hole()
            return None, 0.0, True

        self.reveal_dealer_hole()

        while True:
            dealer_total, _ = hand_value(self.dealer)
            if dealer_total >= 17:
                break
            self.dealer.append(self.draw())

        player_total, _ = hand_value(self.player)
        dealer_total, _ = hand_value(self.dealer)

        if dealer_total > 21 or player_total > dealer_total:
            return None, float(self.bet), True
        if player_total < dealer_total:
            return None, -float(self.bet), True
        return None, 0.0, True
