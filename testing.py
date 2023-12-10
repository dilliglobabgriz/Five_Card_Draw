import random

suits = ['c','d','h','s']
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



    

hand1 = ['2h', '4h', '3h', '6s', '5h']
hand2 = ['7c', '7d', '7h', 'Qs', 'Kh']
hand4 = ['7c', '7d', '5h', '5s', '5h']
hand3 = ['10c', 'Ad', '7h', 'Ks', 'Jh']
hand5 = ['4h', 'Kd', '10h', '7h', 'Ac']

cur_hand = hand3
next_hand = hand5
#print(cur_hand)
print(eval_hand(cur_hand)[0])
print(hand_ranking[eval_hand(cur_hand)[1]])
print(eval_hand(next_hand)[0])
print(hand_ranking[eval_hand(next_hand)[1]])

print(compare_hands(cur_hand, next_hand))