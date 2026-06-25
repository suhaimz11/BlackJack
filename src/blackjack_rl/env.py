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


def draw_card(rng: Random) -> Card:
    """Draw one random card from an infinite deck."""

    cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
    return rng.choice(cards)


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


class BlackjackEnv:
    """A minimal Blackjack environment for learning basic strategy."""

    def __init__(self, seed: int = 7):
        self.rng = Random(seed)
        self.player: list[Card] = []
        self.dealer: list[Card] = []
        self.done = False

    def reset(self) -> State:
        """Start a new hand and return the first state."""

        self.player = [draw_card(self.rng), draw_card(self.rng)]
        self.dealer = [draw_card(self.rng), draw_card(self.rng)]
        self.done = False
        return self.state()

    def state(self) -> State:
        total, usable_ace = hand_value(self.player)
        return State(
            player_total=total,
            dealer_upcard=self.dealer[0],
            usable_ace=usable_ace,
        )

    def legal_actions(self) -> tuple[str, str]:
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
            self.player.append(draw_card(self.rng))
            player_total, _ = hand_value(self.player)

            if player_total > 21:
                self.done = True
                return None, -1.0, True

            return self.state(), 0.0, False

        if action == "stand":
            return self.finish_round()

        raise ValueError(f"Unknown action: {action}")

    def finish_round(self) -> tuple[None, float, bool]:
        """Let the dealer play, then decide win/loss/push."""

        self.done = True

        if is_blackjack(self.player) and not is_blackjack(self.dealer):
            return None, 1.0, True
        if is_blackjack(self.dealer) and not is_blackjack(self.player):
            return None, -1.0, True
        if is_blackjack(self.player) and is_blackjack(self.dealer):
            return None, 0.0, True

        while True:
            dealer_total, _ = hand_value(self.dealer)
            if dealer_total >= 17:
                break
            self.dealer.append(draw_card(self.rng))

        player_total, _ = hand_value(self.player)
        dealer_total, _ = hand_value(self.dealer)

        if dealer_total > 21 or player_total > dealer_total:
            return None, 1.0, True
        if player_total < dealer_total:
            return None, -1.0, True
        return None, 0.0, True
