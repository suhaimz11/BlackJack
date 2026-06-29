from __future__ import annotations

from dataclasses import dataclass
from random import Random


# In this project, a card is stored as its Blackjack value.
# Example: Jack, Queen, King are all stored as 10. Ace starts as 11.
Card = int

CPCS_COUNT_VALUES = {
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 0,
    8: 0,
    9: 0,
    10: -1,
    11: -1,
}


@dataclass(frozen=True)
class State:
    """The information our learning player can see before choosing an action."""

    player_total: int
    dealer_upcard: int
    usable_ace: bool
    high_low_index_bucket: int = 0
    can_double: bool = False
    can_split: bool = False


class Shoe:
    """A finite Blackjack shoe that keeps a CPCS-style running count."""

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
            self.running_count += cpcs_count_value(card)
        return card

    def count_seen_card(self, card: Card) -> None:
        self.running_count += cpcs_count_value(card)

    @property
    def high_low_index(self) -> float:
        unseen_cards = max(len(self.cards), 1)
        return 100 * self.running_count / unseen_cards


def draw_card(rng: Random) -> Card:
    """Draw one random card from an infinite deck."""

    cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
    return rng.choice(cards)


def cpcs_count_value(card: Card) -> int:
    """Return the Complete Point-Count style value of a card."""

    return CPCS_COUNT_VALUES[card]


def index_bucket(high_low_index: float) -> int:
    """Keep the CPCS index state small by rounding and clipping the index."""

    return max(-20, min(20, round(high_low_index)))


def bet_units(high_low_index: float, variable_bet: bool) -> int:
    """Thorp CPCS bet spread based on the high-low index."""

    if not variable_bet:
        return 1
    if high_low_index <= 2:
        return 1
    if high_low_index < 6:
        return 2
    if high_low_index < 8:
        return 3
    if high_low_index < 10:
        return 4
    return 5


def betting_index_band(high_low_index: float) -> str:
    """Return the CPCS betting band used for diagnostics."""

    if high_low_index <= 2:
        return "<=2"
    if high_low_index < 6:
        return "3-5"
    if high_low_index < 8:
        return "6-7"
    if high_low_index < 10:
        return "8-9"
    return ">=10"


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


def can_split_hand(hand: list[Card]) -> bool:
    """Allow one split when the first two cards have the same Blackjack value."""

    return len(hand) == 2 and hand[0] == hand[1]


class BlackjackEnv:
    """A minimal Blackjack environment for learning basic strategy."""

    def __init__(
        self,
        seed: int = 7,
        decks: int | None = None,
        penetration: float = 0.75,
        use_count_in_state: bool = False,
        variable_bet: bool = False,
        dealer_hits_soft_17: bool = False,
        blackjack_payout: float = 1.5,
    ):
        self.rng = Random(seed)
        self.shoe = None if decks is None else Shoe(decks, penetration, self.rng)
        self.use_count_in_state = use_count_in_state
        self.variable_bet = variable_bet
        self.dealer_hits_soft_17 = dealer_hits_soft_17
        self.blackjack_payout = blackjack_payout
        self.player: list[Card] = []
        self.pending_hands: list[tuple[list[Card], int, bool]] = []
        self.finished_hands: list[tuple[list[Card], int]] = []
        self.split_used = False
        self.current_split_aces = False
        self.dealer: list[Card] = []
        self.dealer_hole_counted = False
        self.bet = 1
        self.bet_index_bucket = 0
        self.bet_index_band = "<=2"
        self.done = False

    def draw(self, visible: bool = True) -> Card:
        if self.shoe is None:
            return draw_card(self.rng)
        return self.shoe.draw(update_count=visible)

    def reveal_dealer_hole(self) -> None:
        if self.shoe is not None and not self.dealer_hole_counted:
            self.shoe.count_seen_card(self.dealer[1])
        self.dealer_hole_counted = True

    def current_high_low_index(self) -> float:
        if self.shoe is None:
            return 0.0
        return self.shoe.high_low_index

    def reset(self) -> State:
        """Start a new hand and return the first state."""

        if self.shoe is not None:
            self.shoe.maybe_shuffle()

        betting_index = self.current_high_low_index()
        self.bet = bet_units(betting_index, self.variable_bet)
        self.bet_index_bucket = index_bucket(betting_index) if self.use_count_in_state else 0
        self.bet_index_band = betting_index_band(betting_index) if self.use_count_in_state else "0"
        self.player = [self.draw(), self.draw()]
        self.pending_hands = []
        self.finished_hands = []
        self.split_used = False
        self.current_split_aces = False
        self.dealer = [self.draw(), self.draw(visible=False)]
        self.dealer_hole_counted = False
        self.done = False
        return self.state()

    def reset_to_hard_total(self, player_total: int, dealer_upcard: int) -> State:
        """Start a practice hand from a chosen hard total and dealer upcard."""

        if not 9 <= player_total <= 21:
            raise ValueError("Practice player total must be between 9 and 21.")
        if dealer_upcard not in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            raise ValueError("Dealer upcard must be 2-10 or 11 for ace.")

        # Use only non-ace cards so this is definitely a hard total.
        if player_total <= 19:
            self.player = [2, player_total - 2]
        elif player_total == 20:
            self.player = [10, player_total - 10]
        else:
            self.player = [10, 9, 2]

        self.bet = 1
        self.pending_hands = []
        self.finished_hands = []
        self.split_used = False
        self.current_split_aces = False
        self.dealer = [dealer_upcard, self.draw(visible=False)]
        self.dealer_hole_counted = False
        self.done = False
        return self.state()

    def reset_to_pair(self, pair_card: int, dealer_upcard: int) -> State:
        """Start a practice hand from a chosen pair and dealer upcard."""

        if pair_card not in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            raise ValueError("Pair card must be 2-10 or 11 for aces.")
        if dealer_upcard not in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            raise ValueError("Dealer upcard must be 2-10 or 11 for ace.")

        self.player = [pair_card, pair_card]
        self.bet = 1
        self.pending_hands = []
        self.finished_hands = []
        self.split_used = False
        self.current_split_aces = False
        self.dealer = [dealer_upcard, self.draw(visible=False)]
        self.dealer_hole_counted = False
        self.done = False
        return self.state()

    def reset_to_soft_total(self, soft_total: int, dealer_upcard: int) -> State:
        """Start a practice hand from a chosen soft total and dealer upcard."""

        if not 13 <= soft_total <= 21:
            raise ValueError("Practice soft total must be between 13 and 21.")
        if dealer_upcard not in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            raise ValueError("Dealer upcard must be 2-10 or 11 for ace.")

        self.player = [11, soft_total - 11]
        self.bet = 1
        self.pending_hands = []
        self.finished_hands = []
        self.split_used = False
        self.current_split_aces = False
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
            high_low_index_bucket=index_bucket(self.current_high_low_index()) if self.use_count_in_state else 0,
            can_double=can_double_hand(self.player),
            can_split=can_split_hand(self.player) and not self.split_used,
        )

    def legal_actions(self) -> tuple[str, ...]:
        if self.current_split_aces:
            return ("stand",)

        actions = ["hit", "stand"]
        if can_double_hand(self.player):
            actions.append("double")
        if can_split_hand(self.player) and not self.split_used:
            actions.append("split")
        return tuple(actions)

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
                return self.complete_current_hand()

            return self.state(), 0.0, False

        if action == "double":
            if not can_double_hand(self.player):
                raise ValueError("Double is not legal for this hand.")
            self.bet *= 2
            self.player.append(self.draw())
            player_total, _ = hand_value(self.player)

            if player_total > 21:
                return self.complete_current_hand()

            return self.complete_current_hand()

        if action == "split":
            if not can_split_hand(self.player) or self.split_used:
                raise ValueError("Split is not legal for this hand.")
            first_card, second_card = self.player
            split_aces = first_card == 11
            first_hand = [first_card, self.draw()]
            second_hand = [second_card, self.draw()]
            self.player = first_hand
            self.pending_hands = [(second_hand, self.bet, split_aces)]
            self.current_split_aces = split_aces
            self.split_used = True
            return self.state(), 0.0, False

        if action == "stand":
            return self.complete_current_hand()

        raise ValueError(f"Unknown action: {action}")

    def complete_current_hand(self) -> tuple[State | None, float, bool]:
        """Store the current player hand and continue or finish the round."""

        self.finished_hands.append((self.player, self.bet))

        if self.pending_hands:
            next_hand, next_bet, split_aces = self.pending_hands.pop(0)
            self.player = next_hand
            self.bet = next_bet
            self.current_split_aces = split_aces
            return self.state(), 0.0, False

        return self.finish_round()

    def finish_round(self) -> tuple[None, float, bool]:
        """Let the dealer play, then decide win/loss/push for all player hands."""

        self.done = True

        if not self.finished_hands:
            self.finished_hands.append((self.player, self.bet))

        single_unsplit_hand = len(self.finished_hands) == 1 and not self.split_used

        if single_unsplit_hand and is_blackjack(self.player) and not is_blackjack(self.dealer):
            self.reveal_dealer_hole()
            return None, self.blackjack_payout * self.bet, True
        if single_unsplit_hand and is_blackjack(self.dealer) and not is_blackjack(self.player):
            self.reveal_dealer_hole()
            return None, -float(self.bet), True
        if single_unsplit_hand and is_blackjack(self.player) and is_blackjack(self.dealer):
            self.reveal_dealer_hole()
            return None, 0.0, True

        self.reveal_dealer_hole()

        while True:
            dealer_total, _ = hand_value(self.dealer)
            _, dealer_soft = hand_value(self.dealer)
            if dealer_total > 17:
                break
            if dealer_total == 17 and not (dealer_soft and self.dealer_hits_soft_17):
                break
            self.dealer.append(self.draw())

        dealer_total, _ = hand_value(self.dealer)

        total_reward = 0.0
        for hand, bet in self.finished_hands:
            player_total, _ = hand_value(hand)
            if player_total > 21:
                total_reward -= float(bet)
            elif dealer_total > 21 or player_total > dealer_total:
                total_reward += float(bet)
            elif player_total < dealer_total:
                total_reward -= float(bet)

        return None, total_reward, True
