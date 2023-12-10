import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns


suits = ['c','d','h','s']
suits_to_num = {'c':1, 'd':2, 'h':3, 's':4}
ranks = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K']
ranks_to_num =  {'2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, 'J':11, 'Q':12, 'K':13, 'A':14}
hand_ranking = {
    0:'Royal Flush',
    1:'Straight Flush',
    2:'Four of a Kind',
    3:'Full House',
    4:'Flush',
    5:'Straight',
    6:'Three of a Kind',
    7:'Two Pair',
    8:'Pair',
    9:'High Card'
}

# Variables for csv
player_cards = []
hand_rank = []
c1_list_rank = []
c2_list_rank = []
c3_list_rank = []
c4_list_rank = []
c5_list_rank = []
c1_list_suit = []
c2_list_suit = []
c3_list_suit = []
c4_list_suit = []
c5_list_suit = []
num_players = 4
swap_pos = []
swap_rank = []
swap_suit = []
swapped_hand_rank = []
# Outcome will be 1 if the swapped hand is better than the original hand
# 0 if the hands are the same (just different suits)
# -1 if the original hand is better than the swapped hand
outcome = []

def run_sim(simulations):
    # Function to help sort hands by rank
    def custom_sort(card):
        rank = card[:-1]
        return ranks_to_num[rank]
    
    # Hand eval should take in a string of 5 cards and return a list
    # This list should include the hand sorted and the rank of the hand class
    def eval_hand(hand):
        pair_check =  {'2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0, 'J':0, 'Q':0, 'K':0, 'A':0}
        # Fill pair check with the occurences of each card rank
        for card in hand:
            pair_check[card[:-1]] += 1
        #print(pair_check)
        # Card value list is a the hand without the ranks ('A' -> 14)
        card_values = [ranks_to_num[card[:-1]] for card in hand]

        # Function to sort hands where the cards a specified rank are put first then the rest in descending order A-2
        def poker_sort(rank):
            temp_hand = []
            # Take out the cards in each hand that match the selected rank and put them first
            for card in hand:
                if ranks_to_num[card[:-1]] == rank:
                    temp_hand.append(card)
            # Find the cards that have not been added using set difference
            remaining_cards = list(set(hand) - set(temp_hand))
            # Extend temp hand with the remaining cards sorted by descending order
            temp_hand.extend(sorted(remaining_cards, key=custom_sort, reverse=True))
            # Return the sorted hand
            return temp_hand

        def is_flush():
            # Check for five card of the same suit
            # Suit list keeps track of how many clubs, diamonds, hearts, spades are in hand in that order
            suit_dict = {'c':0, 'd':0, 'h':0, 's':0}
            for card in hand:
                suit_dict[card[-1]] += 1
            for suit in suits:
                if suit_dict[suit] == 5:
                    return True
            return False
        #print(f'Is flush - {is_flush()}')
        def is_straight():
            # Convert card ranks to numerical values
            card_values = [ranks_to_num[card[:-1]] for card in hand]
            # Manually check for A-5 straights (wheel)
            if 2 in card_values and 3 in card_values and 4 in card_values and 5 in card_values and 14 in card_values:
                return 5
            # Check if the values form a consecutive sequence
            card_values.sort()
            for i in range(1, len(card_values)):
                if card_values[i] != card_values[i - 1] + 1:
                    return 0
            return max(card_values)
        #print(f'Is straight - {is_straight()}')

        # Check if a hand has four of a kind
        # Return the rank of the quads if any and 0 otherwise
        def is_quads():
            if (max(pair_check.values()) == 4):
                result = ranks_to_num[(max(pair_check, key=lambda k: pair_check[k]))]
                return result
            return 0
        #print(f'Is quads - {is_quads()}')

        # Check if a hand has three of a kind (and not quads)
        def is_trips():
            if (max(pair_check.values()) == 3):
                return ranks_to_num[(max(pair_check, key=lambda k: pair_check[k]))]
            return 0
        #print(f'Is trips - {is_trips()}')

        # Check for the number of pairs in a hand (not trips or quads)
        # Return the rank of the pair if there is only one pair and the rank of both pairs if there are two
        # Has been replaced
        def num_pairs2():
            max_value = max(pair_check.values())
            if (max_value) == 2:
                key_for_max_value = [key for key, value in pair_check.items() if value == max_value]
                return key_for_max_value
            return False
        #print(f'Number of pairs - {num_pairs()}') 

        # Updated num pairs function that now works properly for diagnosing full houses
        def num_pairs():
            pair_list = [key for key, value in pair_check.items() if value == 2]
            if pair_list == []:
                return 0
            return pair_list

        # Check for hand rank 8 and 9 Straight Flushes
        if (is_flush() and is_straight()):
            if (14 in card_values and 13 in card_values):
                return [sorted(hand), 0]
            return_hand = sorted(hand, key=custom_sort, reverse=True)
            if (is_straight() == 5):
                # For wheel straight flushes move ace to back of list
                ace = return_hand.pop(0)
                return_hand.append(ace)
            return [return_hand, 1]

        # Check for quads
        quad_val = is_quads()
        if (quad_val > 0):
            return [poker_sort(quad_val), 2]

        # Check for full house
        if (is_trips() > 0 and num_pairs() != 0):
            full_val = is_trips()
            return [poker_sort(full_val), 3]
        # Check for flush
        if (is_flush()):
            return [sorted(hand, key=custom_sort, reverse=True), 4]
        # Check for straight
        if (is_straight()):
            return_hand = sorted(hand, key=custom_sort, reverse=True)
            if (is_straight() == 5):
                # For wheel straights the ace should be at the end not the front
                ace = return_hand.pop(0)
                return_hand.append(ace)
            return [return_hand, 5]

        # Check for trips
        trip_val = is_trips()
        if (trip_val > 0):
            return [poker_sort(trip_val), 6]

        # Check for two pair
        if(num_pairs()):
            if (len(num_pairs()) == 2):
                minc, maxc = num_pairs()
                minc = ranks_to_num[minc]
                maxc = ranks_to_num[maxc]
                temp_hand = []
                # Add the highest pair to the sorted hand first
                for c1 in hand:
                    if ranks_to_num[c1[:-1]] == maxc:
                        temp_hand.append(c1)
                # Then add the other pair to the sorted hand
                for c2 in hand:
                    if ranks_to_num[c2[:-1]] == minc:
                        temp_hand.append(c2)
                # Finally add the remaining card to the hand
                for c3 in hand:
                    if ranks_to_num[c3[:-1]] != maxc and ranks_to_num[c3[:-1]] != minc:
                        temp_hand.append(c3)
                return [temp_hand, 7] 

        # Check for pair
        if (num_pairs()):
            pair_val = ranks_to_num[num_pairs()[0]]
            return [poker_sort(pair_val), 8]
    
        # Default case is high card
        return [sorted(hand, key=custom_sort, reverse=True), 9]
    
    
    # Compare hands function will be used for generating the outcome
    # It takes as input two five card hands and returns either a 1 or 0
    # 0 means the first hand is either better or the same as the second hand
    # 1 means the second hand is better
    def compare_hands(h1, h2):
        h1_eval = eval_hand(h1)
        h2_eval = eval_hand(h2)
        
        # If the class rank of one hand is lower than the lower class hand is better
        if h1_eval[1] < h2_eval[1]:
            return 0
        elif h1_eval[1] > h2_eval[1]:
            return 1
        # If the hands are of the same class we should go card by card through the hand comparing the ranks of each card in the sorted hands
        else:
            for i in range(5):
                if ranks_to_num[h1_eval[0][i][:-1]] > ranks_to_num[h2_eval[0][i][:-1]]:
                    return 0
                elif ranks_to_num[h1_eval[0][i][:-1]] < ranks_to_num[h2_eval[0][i][:-1]]:
                    return 1
        # If everything is the same return 0
        # Trying using binary options instead
        return 0


    for i in range(simulations):
        # Create deck function to create and shuffle a new deck
        def create_deck():
            temp_deck = []
            for suit in suits:
                for rank in ranks:
                    temp_deck.append(f'{rank}{suit}')   
            random.shuffle(temp_deck)
            return temp_deck
        
        deck = create_deck()
        player_hands = [[] for _ in range(num_players)]
        
        # Deal 5 cards for each player
        for i in range(5):
            for player in player_hands:
                player.append(deck.pop())
        
        # Choose a card to swap based on the strength of the hand
        # Should return a zero if the best decision is to not swap, and 1-5 if the card should be swapped
        def choose_card(hand):
            # Choices of cards to swap
            choices = [0,1,2,3,4,5]
            weights1 = [0.95, 0.01, 0.01, 0.01, 0.01, 0.01]
            weights2 = [0, 0, 0, 0, .2, .8]
            weights3 = [0, .05, .05, .1, .3, .5]
            weights4 = [0, .05, .15, .15, .15, .5]
            # srt represents the sorted hand and rnk is the numerical rank of the hand
            srt, rnk = eval_hand(hand)
            # Hand better than a straight will not be worth swapping and risking breaking the hand
            if (rnk < 6):
                return random.choices(choices, weights=weights1, k=1)[0]
            # For trips it makes sense to swap the lowest card in the hand to try to improve to a full house or quads
            if (rnk == 6):
                return random.choices(choices, weights=weights2, k=1)[0]
            # For two pair return the unpaired card
            if (rnk == 7):
                return 5
            # For single paired hand swap one of the unpaired cards most of the time
            if (rnk == 8):
                return random.choices(choices, weights=weights3, k=1)[0]
            # For unpaired hand swap one card randomly
            # Swap the lowest rank card more than the highest rank card
            if (rnk == 9):
                return random.choices(choices, weights=weights4, k=1)[0]


        for player in player_hands:
            swap_choice = choose_card(player)
            # Pop one card of the deck, this will be swapped in each position
            swap_card = deck.pop()

            # List updates for the hand ranks, hand suits, and overall hand rank
            # Formatted ranks and suits to be two seperate integers for ML model
            hand_list = eval_hand(player)[0]
            c1, c2, c3, c4, c5 = hand_list
            c1_list_rank.append(ranks_to_num[c1[:-1]])
            c2_list_rank.append(ranks_to_num[c2[:-1]])
            c3_list_rank.append(ranks_to_num[c3[:-1]])
            c4_list_rank.append(ranks_to_num[c4[:-1]])
            c5_list_rank.append(ranks_to_num[c5[:-1]])
    
            c1_list_suit.append(suits_to_num[c1[-1]])
            c2_list_suit.append(suits_to_num[c2[-1]])
            c3_list_suit.append(suits_to_num[c3[-1]])
            c4_list_suit.append(suits_to_num[c4[-1]])
            c5_list_suit.append(suits_to_num[c5[-1]])
            hand_rank.append(eval_hand(player)[1])
            
            temp_hand = list(player)
            # Sort temp hand into proper order before swapping
            temp_hand = eval_hand(temp_hand)[0]
            if (swap_choice != 0):
                temp_hand[swap_choice-1] = swap_card

            #temp_hand = eval_hand(temp_hand)[0]
            # Find out if the swapped hand is better and update the outcome
            #find_outcome = compare_hands(temp_hand, player)
            #print(f'og hand: {player} \nswapped hand: {temp_hand}')
            find_outcome = compare_hands(player, temp_hand)
            #print(f'{find_outcome}\n')
            
            # Update the swap and new rank information for the csv
            swap_pos.append(swap_choice)
            swap_rank.append(ranks_to_num[swap_card[:-1]])
            swap_suit.append(suits_to_num[swap_card[-1]])
            swapped_hand_rank.append(eval_hand(temp_hand)[1])
            outcome.append(find_outcome)
            # Determine the label
            # If the original hand is better than or the same as the swapped hand the label should be 0
            # If the swapped hand beats the original hand the label should be 1
            # Since no change occurs if there is no swap, this label should always be 0
            '''
            for i in range(5):
                # List updates for the hand ranks, hand suits, and overall hand rank
                # Formatted ranks and suits to be two seperate integers for ML model
                hand_list = eval_hand(player)[0]
                c1, c2, c3, c4, c5 = hand_list

                c1_list_rank.append(ranks_to_num[c1[:-1]])
                c2_list_rank.append(ranks_to_num[c2[:-1]])
                c3_list_rank.append(ranks_to_num[c3[:-1]])
                c4_list_rank.append(ranks_to_num[c4[:-1]])
                c5_list_rank.append(ranks_to_num[c5[:-1]])
        
                c1_list_suit.append(suits_to_num[c1[-1]])
                c2_list_suit.append(suits_to_num[c2[-1]])
                c3_list_suit.append(suits_to_num[c3[-1]])
                c4_list_suit.append(suits_to_num[c4[-1]])
                c5_list_suit.append(suits_to_num[c5[-1]])
                hand_rank.append(eval_hand(player)[1])

                # Create a copy of the original hand to do modifications on 
                temp_hand = list(player)
                # Swap a card from the deck to the card in position i
                temp_hand[i] = swap_card
                # Find out if the swapped hand is better and update the outcome
                find_outcome = compare_hands(temp_hand, player)
                # Update the swap and new rank information for the csv
                swap_pos.append(i+1)
                swap_rank.append(ranks_to_num[swap_card[:-1]])
                swap_suit.append(suits_to_num[swap_card[-1]])
                swapped_hand_rank.append(eval_hand(temp_hand)[1])
                outcome.append(find_outcome)
                '''
        
    df = pd.DataFrame()    
    df['c1_rank'] = c1_list_rank
    df['c2_rank'] = c2_list_rank
    df['c3_rank'] = c3_list_rank
    df['c4_rank'] = c4_list_rank
    df['c5_rank'] = c5_list_rank
    df['c1_suit'] = c1_list_suit
    df['c2_suit'] = c2_list_suit
    df['c3_suit'] = c3_list_suit
    df['c4_suit'] = c4_list_suit
    df['c5_suit'] = c5_list_suit
    df['class'] = hand_rank
    # s for swap
    df['s_pos'] = swap_pos
    df['s_rank'] = swap_rank
    df['s_suit'] = swap_suit
    df['s_class'] = swapped_hand_rank
    df['outcome'] = outcome
    print(df.describe())
    df.to_csv('fivecarddraw_400k.csv')

run_sim(100000)



