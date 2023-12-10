# Load the ML model for 5 card draw and create a function to allow me to input a hand, swap position, and hand class
# Model will return a result that determines if this card should be swapped or not

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout
import tensorflow as tf

# Modify this hand and press run to get the best card to swap for the given hand
#p0 = ['Qh', 'Kh', 'Qs', 'Qc', 'Kd']
p0 = ['9h', '8h', '7h', '6h', 'Kc']
h8 = ['Qh', 'Kh', '5s', 'Qc', '10d']
h7 = ['10h', '5h', '5s', '2c', '10d']
h6 = ['2h', 'Kh', '5s', '2c', '2d']
h5 = ['Qh', 'Kh', 'Js', '9c', '10d']
h4 = ['Qh', 'Kh', '5h', '2h', '10h']
h3 = ['3h', '3d', '3s', '10c', '10d']
h2 = ['7h', '7h', '7s', '7c', 'Ad']
h1 = ['5h', '4h', '3h', '2h', 'Ah']
h0 = ['As', 'Ks', 'Qs', 'Js', '10s']


suits = ['c','d','h','s']
suits_to_num = {'c':1, 'd':2, 'h':3, 's':4}
ranks = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K']
ranks_to_num =  {'2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, 'J':11, 'Q':12, 'K':13, 'A':14}

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

# Load the model
new_model = tf.keras.models.load_model('basic_model.keras')

# Check that the information in the model is what we are looking for
#print(new_model.summary())

# Model decision function
def model_decision(model, c1_rank, c2_rank, c3_rank, c4_rank, c5_rank, c1_suit, c2_suit, c3_suit, c4_suit, c5_suit, hand_class, s_pos):
    # The output is what the correct decision is based on the current hand and the card that is being swapped (swap = 1, stay = 0)

    d = {'c1_rank':[c1_rank], 'c2_rank':[c2_rank], 'c3_rank':[c3_rank], 'c4_rank':[c4_rank], 'c5_rank':[c5_rank], 
         'c1_suit':[c1_suit], 'c2_suit':[c2_suit], 'c3_suit':[c3_suit], 'c4_suit':[c4_suit], 'c5_suit':[c5_suit],
         'class':[hand_class], 's_pos':[s_pos]}
    
    # This dataframe is what will be run through the model
    input_df = pd.DataFrame(data=d)

    # Running data through model
    prediction = model.predict(input_df)

    return prediction
    # Set the threshold for when to stay or swap
    if prediction > 0.5:
        return 1
    else:
        return 0
    
# Function that should take a hand and return the best card to swap 0-5
def best_swap(hand):
    hand = eval_hand(hand)[0]
    # First parse the hand so model decision can be used
    c1_rank = ranks_to_num[hand[0][:-1]]
    c2_rank = ranks_to_num[hand[1][:-1]]
    c3_rank = ranks_to_num[hand[2][:-1]]
    c4_rank = ranks_to_num[hand[3][:-1]]
    c5_rank = ranks_to_num[hand[4][:-1]]
    c1_suit = suits_to_num[hand[0][-1]]
    c2_suit = suits_to_num[hand[1][-1]]
    c3_suit = suits_to_num[hand[2][-1]]
    c4_suit = suits_to_num[hand[3][-1]]
    c5_suit = suits_to_num[hand[4][-1]]
    hand_class = eval_hand(hand)[1]
    model_outputs = []
    for i in range(5):
        model_outputs.append(model_decision(new_model, c1_rank, c2_rank, c3_rank, c4_rank, c5_rank, c1_suit, c2_suit, c3_suit, c4_suit, c5_suit, hand_class, i+1))
    if max(model_outputs) < .4:
        return 0
    else:
        return (model_outputs.index(max(model_outputs)) + 1)

# Testing with actual hands
action = ['stay', 'swap']
'''
# Should stay on straight usually
print(action[model_decision(new_model, 11, 10, 9, 8, 7, 1, 1, 1, 2, 2, 5, 5)])
print('STAY')
print(action[model_decision(new_model, 6, 5, 4, 3, 2, 1, 3, 1, 2, 2, 5, 4)])
print('STAY')
# Should always swap just high card hand
print(action[model_decision(new_model, 11, 10, 6, 5, 4, 1, 1, 1, 2, 2, 9, 5)])
print('SWAP')
# Should stay with sf
print(action[model_decision(new_model, 11, 10, 9, 8, 7, 1, 1, 1, 1, 1, 1, 5)])
print('STAY')
# With two pair should swap card 5 but not card 4
print(action[model_decision(new_model, 11, 11, 9, 9, 7, 1, 2, 1, 2, 1, 7, 5)])
print('SWAP')
print(action[model_decision(new_model, 11, 11, 9, 9, 7, 1, 2, 1, 2, 1, 7, 4)])
print('STAY')
print(action[model_decision(new_model, 11, 11, 9, 8, 7, 1, 2, 1, 2, 1, 8, 5)])
print('SWAP')
print(action[model_decision(new_model, 11, 11, 9, 8, 7, 1, 2, 1, 2, 1, 8, 4)])
print('SWAP')
print(action[model_decision(new_model, 11, 11, 9, 8, 7, 1, 2, 1, 2, 1, 8, 2)])
print('?')
'''

#print(action[model_decision(new_model, 14, 14, 14, 13, 5, 4, 1, 2, 3, 1, 6, 5)])
#print('AAAK5 decision')

#print(action[model_decision(new_model, 7, 7, 7, 12, 12, 3, 2, 1, 1, 4, 3, 5)])
#print('777QQ decision')

print(f'Given the hand: {eval_hand(p0)[0]}\nThe best card to swap is the card in position: {best_swap(p0)}')