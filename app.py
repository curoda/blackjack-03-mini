import random
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------
# Global Definitions and Helpers
# -----------------------------
# Card values (10, J, Q, K all count as 10).
CARD_VALUES = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 11
}

# List of card ranks.
RANKS = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']

def initialize_shoe(num_decks=8):
    """
    Create an 8‑deck shoe.
    Each deck has 4 copies of each rank.
    """
    shoe = {}
    for rank in RANKS:
        shoe[rank] = 4 * num_decks
    return shoe

def total_cards(shoe):
    return sum(shoe.values())

def draw_card(shoe):
    """
    Draw a random card from the shoe, weighted by remaining counts.
    Returns the drawn card and the updated shoe.
    """
    total = total_cards(shoe)
    r = random.randint(1, total)
    cumulative = 0
    for card, count in shoe.items():
        cumulative += count
        if r <= cumulative:
            shoe[card] -= 1
            return card, shoe
    # Fallback (should never occur)
    return None, shoe

def calculate_hand_value(hand):
    """
    Compute the best total for a hand.
    Counts Aces as 11 unless that would cause a bust.
    """
    total = 0
    aces = 0
    for card in hand:
        total += CARD_VALUES[card]
        if card == 'A':
            aces += 1
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total

def is_soft(hand):
    """
    Return True if the hand is "soft" (i.e., contains an Ace counted as 11).
    A hand is soft if the raw total (assuming every Ace is 11) is equal to the best total
    computed by calculate_hand_value (meaning no Ace was reduced).
    """
    raw_total = sum(CARD_VALUES[card] for card in hand)
    best_total = calculate_hand_value(hand)
    return ('A' in hand) and (raw_total == best_total)

def is_pair(hand):
    """
    Return True if the hand is a pair (only two cards and of identical rank).
    (For simulation, note that a 10 and a King are not considered a pair.)
    """
    return len(hand) == 2 and hand[0] == hand[1]

# -----------------------------
# Perfect Basic Strategy Decision Logic
# -----------------------------
def basic_strategy_decision(hand, dealer_up, first_move=True):
    """
    Given the player's hand (list of cards) and the dealer's upcard (string),
    return one of: "stand", "hit", "double", "split", or "surrender".
    
    Assumptions:
      - Surrender and doubling are allowed only on the first move.
      - Splitting is allowed only if the hand is a pair.
      
    The rules below reflect a commonly accepted basic–strategy chart for an 8‐deck game,
    dealer hits on soft 17, and surrender allowed.
    """
    total = calculate_hand_value(hand)
    soft = is_soft(hand)
    
    # --- Surrender (only on first move) ---
    if first_move and len(hand) == 2 and not soft:
        if total == 16 and dealer_up in ['9','10','J','Q','K','A']:
            return "surrender"
        if total == 15 and dealer_up in ['10','J','Q','K','A']:
            return "surrender"
    
    # --- Splitting (only on pairs) ---
    if is_pair(hand) and first_move:
        card = hand[0]
        # Always split Aces and 8's.
        if card == 'A' or card == '8':
            return "split"
        # Never split tens.
        if card in ['10','J','Q','K']:
            return "stand"
        # Split 2's and 3's if dealer upcard is 2–7.
        if card in ['2', '3']:
            if dealer_up in ['2','3','4','5','6','7']:
                return "split"
            else:
                return "hit"
        # Split 4's if dealer upcard is 5 or 6.
        if card == '4':
            if dealer_up in ['5','6']:
                return "split"
            else:
                return "hit"
        # Pair of 5's: never split (treat as a hard 10).
        if card == '5':
            pass  # Let the below logic for hard totals handle it.
        # Split 6's if dealer upcard is 2–6.
        if card == '6':
            if dealer_up in ['2','3','4','5','6']:
                return "split"
            else:
                return "hit"
        # Split 7's if dealer upcard is 2–7.
        if card == '7':
            if dealer_up in ['2','3','4','5','6','7']:
                return "split"
            else:
                return "hit"
        # Split 9's if dealer upcard is 2–6 or 8–9; otherwise stand.
        if card == '9':
            if dealer_up in ['2','3','4','5','6','8','9']:
                return "split"
            else:
                return "stand"
    
    # --- Doubling Down (only on first move) ---
    if first_move and len(hand) == 2:
        if not soft:
            if total == 11:
                return "double"
            if total == 10 and dealer_up not in ['10','J','Q','K','A']:
                return "double"
            if total == 9 and dealer_up in ['3','4','5','6']:
                return "double"
        else:
            # Soft totals.
            if total in [13, 14] and dealer_up in ['5','6']:
                return "double"
            if total in [15, 16] and dealer_up in ['4','5','6']:
                return "double"
            if total == 17 and dealer_up in ['3','4','5','6']:
                return "double"
            if total == 18 and dealer_up in ['3','4','5','6']:
                return "double"
    
    # --- Standing vs. Hitting ---
    if not soft:
        if total >= 17:
            return "stand"
        if total >= 13 and dealer_up in ['2','3','4','5','6']:
            return "stand"
        if total == 12 and dealer_up in ['4','5','6']:
            return "stand"
        return "hit"
    else:
        # Soft hands.
        if total >= 19:
            return "stand"
        if total == 18:
            if dealer_up in ['9','10','J','Q','K','A']:
                return "hit"
            else:
                return "stand"
        return "hit"

# -----------------------------
# Player Hand Simulation (Recursive to Handle Splits)
# -----------------------------
def play_player_hand(hand, dealer_up, shoe, first_move=True, is_split_aces=False):
    """
    Simulate playing a player's hand using perfect basic strategy.
    
    Handles:
      - Surrender (if chosen on the first move)
      - Doubling down (take one card then stand)
      - Splitting (recursive call for each split hand)
      - Standard hit/stand decisions
    
    Parameters:
      hand: current hand (list of cards)
      dealer_up: dealer’s upcard (string)
      shoe: current shoe (dictionary)
      first_move: True if this is the first decision for the hand
      is_split_aces: True if the hand comes from a split of aces (only one card drawn)
    
    Returns a tuple (hands_results, shoe, cards_used) where hands_results is a list
    of dictionaries. Each dictionary represents a final hand with keys:
      - 'hand': final list of cards
      - 'bet': bet multiplier (1 normally, 2 if doubled)
      - 'surrender': True if surrendered (results in -0.5 loss)
      - 'blackjack': True if a natural blackjack (only possible on first move)
      - 'busted': True if the hand busts
      - 'total': final total (if not busted)
    cards_used counts the number of cards drawn in playing this hand.
    """
    hands_results = []
    cards_used = 0

    # Check for natural blackjack (only allowed on first move, not from splits of aces)
    if first_move and len(hand) == 2 and calculate_hand_value(hand) == 21 and not is_split_aces:
        hands_results.append({
            'hand': hand,
            'bet': 1,
            'surrender': False,
            'blackjack': True,
            'busted': False,
            'total': 21
        })
        return hands_results, shoe, cards_used

    # Special rule: for a hand of split aces, draw one card and stand.
    if is_split_aces:
        card, shoe = draw_card(shoe)
        cards_used += 1
        new_hand = hand + [card]
        total = calculate_hand_value(new_hand)
        hands_results.append({
            'hand': new_hand,
            'bet': 1,
            'surrender': False,
            'blackjack': False,
            'busted': total > 21,
            'total': total
        })
        return hands_results, shoe, cards_used

    # Decide action based on basic strategy.
    decision = basic_strategy_decision(hand, dealer_up, first_move)

    if decision == "surrender":
        hands_results.append({
            'hand': hand,
            'bet': 1,
            'surrender': True,
            'blackjack': False,
            'busted': False,
            'total': calculate_hand_value(hand)
        })
        return hands_results, shoe, cards_used

    elif decision == "split":
        # For splitting, remove the pair and create two hands.
        card_to_split = hand[0]
        results_all = []
        for i in range(2):
            new_hand = [card_to_split]
            card, shoe = draw_card(shoe)
            cards_used += 1
            new_hand.append(card)
            split_aces = (card_to_split == 'A')
            res, shoe, used = play_player_hand(new_hand, dealer_up, shoe, first_move=True, is_split_aces=split_aces)
            cards_used += used
            results_all.extend(res)
        return results_all, shoe, cards_used

    elif decision == "double":
        # For doubling, take exactly one card then stand.
        card, shoe = draw_card(shoe)
        cards_used += 1
        new_hand = hand + [card]
        total = calculate_hand_value(new_hand)
        hands_results.append({
            'hand': new_hand,
            'bet': 2,
            'surrender': False,
            'blackjack': False,
            'busted': total > 21,
            'total': total
        })
        return hands_results, shoe, cards_used

    elif decision == "stand":
        total = calculate_hand_value(hand)
        hands_results.append({
            'hand': hand,
            'bet': 1,
            'surrender': False,
            'blackjack': False,
            'busted': total > 21,
            'total': total
        })
        return hands_results, shoe, cards_used

    elif decision == "hit":
        card, shoe = draw_card(shoe)
        cards_used += 1
        new_hand = hand + [card]
        total = calculate_hand_value(new_hand)
        if total > 21:
            hands_results.append({
                'hand': new_hand,
                'bet': 1,
                'surrender': False,
                'blackjack': False,
                'busted': True,
                'total': total
            })
            return hands_results, shoe, cards_used
        else:
            # Continue playing; now not the first move.
            return play_player_hand(new_hand, dealer_up, shoe, first_move=False, is_split_aces=False)
    else:
        # Fallback
        total = calculate_hand_value(hand)
        hands_results.append({
            'hand': hand,
            'bet': 1,
            'surrender': False,
            'blackjack': False,
            'busted': total > 21,
            'total': total
        })
        return hands_results, shoe, cards_used

# -----------------------------
# Dealer Hand Simulation
# -----------------------------
def play_dealer_hand(hand, shoe):
    """
    Simulate dealer play.
    Dealer hits until total >= 17, and hits on soft 17.
    Returns (final_hand, updated_shoe, cards_used).
    """
    cards_used = 0
    while True:
        total = calculate_hand_value(hand)
        if total < 17 or (total == 17 and is_soft(hand)):
            card, shoe = draw_card(shoe)
            cards_used += 1
            hand.append(card)
        else:
            break
    return hand, shoe, cards_used

# -----------------------------
# Simulating a Single Round of Blackjack
# -----------------------------
def simulate_hand(shoe):
    """
    Simulate one round of blackjack:
      - Deal initial cards for player and dealer.
      - Check for natural blackjack.
      - Play out the player's hand(s) (including splits, doubles, surrender).
      - Play out the dealer's hand.
      - Determine outcome for each player hand.
    
    Returns:
      outcome: net monetary result for the round (summed over player hands)
      cards_used: total number of cards drawn in the round
      updated_shoe: updated shoe after the round.
    """
    cards_used = 0

    # Deal initial cards.
    player_hand = []
    dealer_hand = []
    for _ in range(2):
        card, shoe = draw_card(shoe)
        cards_used += 1
        player_hand.append(card)
    for _ in range(2):
        card, shoe = draw_card(shoe)
        cards_used += 1
        dealer_hand.append(card)
    dealer_up = dealer_hand[0]

    # Check for natural blackjack.
    player_blackjack = (len(player_hand) == 2 and calculate_hand_value(player_hand) == 21)
    dealer_blackjack = (len(dealer_hand) == 2 and calculate_hand_value(dealer_hand) == 21)
    if player_blackjack or dealer_blackjack:
        if player_blackjack and not dealer_blackjack:
            return 1.5, cards_used, shoe
        elif dealer_blackjack and not player_blackjack:
            return -1.0, cards_used, shoe
        else:
            return 0.0, cards_used, shoe

    # Play out the player's hand(s).
    player_hands, shoe, used = play_player_hand(player_hand, dealer_up, shoe, first_move=True)
    cards_used += used

    # Dealer plays.
    dealer_hand, shoe, used_dealer = play_dealer_hand(dealer_hand, shoe)
    cards_used += used_dealer
    dealer_total = calculate_hand_value(dealer_hand)
    dealer_bust = dealer_total > 21

    # Determine outcome for each player hand.
    round_outcome = 0.0
    for ph in player_hands:
        bet = ph['bet']
        if ph['surrender']:
            round_outcome += -0.5 * bet
        elif ph['blackjack']:
            round_outcome += 1.5 * bet
        elif ph['busted']:
            round_outcome += -1.0 * bet
        else:
            player_total = ph['total']
            if dealer_bust:
                round_outcome += 1.0 * bet
            else:
                if player_total > dealer_total:
                    round_outcome += 1.0 * bet
                elif player_total == dealer_total:
                    round_outcome += 0.0
                else:
                    round_outcome += -1.0 * bet
    return round_outcome, cards_used, shoe

# -----------------------------
# Shoe Management & Session Simulation
# -----------------------------
def reshuffle_if_needed(shoe, cards_dealt, penetration=0.75, num_decks=8):
    """
    Reshuffle the shoe if the remaining cards drop below the threshold.
    """
    if total_cards(shoe) < (num_decks * 52) * (1 - penetration):
        return initialize_shoe(num_decks), 0
    return shoe, cards_dealt

def simulate_session(profit_target, loss_threshold, max_hands=200, penetration=0.75, num_decks=8):
    """
    Simulate one blackjack session:
      - Session stops if cumulative profit reaches/exceeds profit_target,
        if cumulative loss reaches/exceeds loss_threshold,
        or after max_hands.
    
    Returns a dict:
      - 'profit': final cumulative profit,
      - 'hands_played': number of hands played,
      - 'stop_reason': which condition stopped the session.
    """
    cumulative_profit = 0.0
    hands_played = 0
    shoe = initialize_shoe(num_decks)
    cards_dealt = 0

    while hands_played < max_hands:
        shoe, cards_dealt = reshuffle_if_needed(shoe, cards_dealt, penetration, num_decks)
        outcome, used, shoe = simulate_hand(shoe)
        cumulative_profit += outcome
        hands_played += 1
        cards_dealt += used

        if cumulative_profit >= profit_target:
            stop_reason = 'profit_target'
            break
        elif cumulative_profit <= -loss_threshold:
            stop_reason = 'loss_threshold'
            break
    else:
        stop_reason = 'max_hands'
    return {
        'profit': cumulative_profit,
        'hands_played': hands_played,
        'stop_reason': stop_reason
    }

# -----------------------------
# Optimization Functions
# -----------------------------
def run_simulation_for_params(profit_target, loss_threshold, num_sessions=1000):
    """
    Run many sessions for given profit_target and loss_threshold.
    Returns a DataFrame of session results.
    """
    results = []
    for _ in range(num_sessions):
        session_result = simulate_session(profit_target, loss_threshold)
        results.append(session_result)
    return pd.DataFrame(results)

def run_optimization(profit_targets, loss_thresholds, num_sessions=1000):
    """
    Run simulations for each combination of profit target and loss threshold.
    
    Returns a DataFrame summarizing, for each parameter combination:
      - prob_profit_target: Fraction of sessions stopping because the profit target was reached.
      - prob_loss_threshold: Fraction of sessions stopping because the loss threshold was reached.
      - prob_win_non_target: Fraction of sessions that ended at max hands with a win (< profit_target).
      - prob_loss_non_threshold: Fraction of sessions that ended at max hands with a loss (but not hitting loss threshold).
      - avg_hands: Average number of hands played.
      - exp_profit: Overall expected profit.
    """
    optimization_results = []
    for pt in profit_targets:
        for lt in loss_thresholds:
            df = run_simulation_for_params(pt, lt, num_sessions)
            # Outcome categories:
            prob_profit_target = (df['stop_reason'] == 'profit_target').mean()
            prob_loss_threshold = (df['stop_reason'] == 'loss_threshold').mean()
            # For sessions ending by max hands, further classify by profit sign:
            max_sessions = df[df['stop_reason'] == 'max_hands']
            if len(max_sessions) > 0:
                prob_win_non_target = (max_sessions['profit'] > 0).mean()
                prob_loss_non_threshold = (max_sessions['profit'] < 0).mean()
            else:
                prob_win_non_target = 0.0
                prob_loss_non_threshold = 0.0

            avg_hands = df['hands_played'].mean()
            exp_profit = df['profit'].mean()
            optimization_results.append({
                'profit_target': pt,
                'loss_threshold': lt,
                'prob_profit_target': prob_profit_target,
                'prob_loss_threshold': prob_loss_threshold,
                'prob_win_non_target': prob_win_non_target,
                'prob_loss_non_threshold': prob_loss_non_threshold,
                'avg_hands': avg_hands,
                'exp_profit': exp_profit
            })
    return pd.DataFrame(optimization_results)

# -----------------------------
# Streamlit Front End
# -----------------------------
def main():
    st.title("Blackjack Session Optimization Simulator (Perfect Basic Strategy)")
    st.markdown("""
    This simulator runs blackjack sessions using an 8‑deck shoe (75% penetration) and implements perfect basic strategy,
    including surrender, double downs, and unlimited splits (with split aces receiving only one card).

    Each session stops when the profit target is reached, the loss threshold is hit, or after 200 hands.
    Adjust the parameters below and click **Run Simulation**.
    """)

    st.sidebar.header("Simulation Parameters")
    profit_target_min = st.sidebar.number_input("Minimum Profit Target (units)", value=3, min_value=1)
    profit_target_max = st.sidebar.number_input("Maximum Profit Target (units)", value=20, min_value=profit_target_min)
    loss_threshold_min = st.sidebar.number_input("Minimum Loss Threshold (units)", value=10, min_value=1)
    loss_threshold_max = st.sidebar.number_input("Maximum Loss Threshold (units)", value=20, min_value=loss_threshold_min)
    num_sessions = st.sidebar.number_input("Sessions per Parameter Combination", value=1000, min_value=100, step=100)

    profit_targets = list(range(int(profit_target_min), int(profit_target_max) + 1))
    loss_thresholds = list(range(int(loss_threshold_min), int(loss_threshold_max) + 1))

    if st.sidebar.button("Run Simulation"):
        st.write("Running simulation. Please wait ...")
        optimization_results = run_optimization(profit_targets, loss_thresholds, num_sessions)
        st.write("### Optimization Results")
        st.dataframe(optimization_results)

        # Create pivot tables for each outcome category.
        pivot_profit_target = optimization_results.pivot(index='loss_threshold', columns='profit_target', values='prob_profit_target')
        pivot_win_non_target = optimization_results.pivot(index='loss_threshold', columns='profit_target', values='prob_win_non_target')
        pivot_loss_non_threshold = optimization_results.pivot(index='loss_threshold', columns='profit_target', values='prob_loss_non_threshold')
        pivot_loss_threshold = optimization_results.pivot(index='loss_threshold', columns='profit_target', values='prob_loss_threshold')

        st.write("### Heatmap: Probability of Reaching Profit Target")
        st.dataframe(pivot_profit_target)
        fig1, ax1 = plt.subplots()
        cax1 = ax1.imshow(pivot_profit_target.values, cmap='viridis', origin='lower', aspect='auto')
        ax1.set_xticks(np.arange(len(pivot_profit_target.columns)))
        ax1.set_xticklabels(pivot_profit_target.columns)
        ax1.set_yticks(np.arange(len(pivot_profit_target.index)))
        ax1.set_yticklabels(pivot_profit_target.index)
        ax1.set_xlabel("Profit Target (units)")
        ax1.set_ylabel("Loss Threshold (units)")
        fig1.colorbar(cax1)
        st.pyplot(fig1)

        st.write("### Heatmap: Probability of Wins (Max Hands, No Profit Target)")
        st.dataframe(pivot_win_non_target)
        fig2, ax2 = plt.subplots()
        cax2 = ax2.imshow(pivot_win_non_target.values, cmap='viridis', origin='lower', aspect='auto')
        ax2.set_xticks(np.arange(len(pivot_win_non_target.columns)))
        ax2.set_xticklabels(pivot_win_non_target.columns)
        ax2.set_yticks(np.arange(len(pivot_win_non_target.index)))
        ax2.set_yticklabels(pivot_win_non_target.index)
        ax2.set_xlabel("Profit Target (units)")
        ax2.set_ylabel("Loss Threshold (units)")
        fig2.colorbar(cax2)
        st.pyplot(fig2)

        st.write("### Heatmap: Probability of Losses (Max Hands, No Loss Threshold)")
        st.dataframe(pivot_loss_non_threshold)
        fig3, ax3 = plt.subplots()
        cax3 = ax3.imshow(pivot_loss_non_threshold.values, cmap='viridis', origin='lower', aspect='auto')
        ax3.set_xticks(np.arange(len(pivot_loss_non_threshold.columns)))
        ax3.set_xticklabels(pivot_loss_non_threshold.columns)
        ax3.set_yticks(np.arange(len(pivot_loss_non_threshold.index)))
        ax3.set_yticklabels(pivot_loss_non_threshold.index)
        ax3.set_xlabel("Profit Target (units)")
        ax3.set_ylabel("Loss Threshold (units)")
        fig3.colorbar(cax3)
        st.pyplot(fig3)

        st.write("### Heatmap: Probability of Reaching Loss Threshold")
        st.dataframe(pivot_loss_threshold)
        fig4, ax4 = plt.subplots()
        cax4 = ax4.imshow(pivot_loss_threshold.values, cmap='viridis', origin='lower', aspect='auto')
        ax4.set_xticks(np.arange(len(pivot_loss_threshold.columns)))
        ax4.set_xticklabels(pivot_loss_threshold.columns)
        ax4.set_yticks(np.arange(len(pivot_loss_threshold.index)))
        ax4.set_yticklabels(pivot_loss_threshold.index)
        ax4.set_xlabel("Profit Target (units)")
        ax4.set_ylabel("Loss Threshold (units)")
        fig4.colorbar(cax4)
        st.pyplot(fig4)

if __name__ == '__main__':
    main()
