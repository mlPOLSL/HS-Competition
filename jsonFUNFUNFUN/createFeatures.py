import json
import numpy as np
import math

coefficients = {
    'minions': {
        'shield': 1.407,
        'windfury': 1.195,
        'freezing': 1.027,
        'stealth': 0.604,
        'attack': 0.571,
        'taunt': 0.512,
        'hp': 0.407,
        'charge': 0.327,
        'forgetful': -0.2
    },
    'intrinsic value': -0.357,
    'weapon_durability': 0.591,
    'can_attack': 1.1,
    'frozen': -0.4
}


def estimate_hand_value(hand):
    hand_value = 0
    for card in hand:
        card_value = 0
        if card['type'] == 'MINION':
            for attribute in coefficients['minions']:
                if attribute == 'charge':
                    card_value += float(card[attribute]) * float(card['attack']) * coefficients['minions'][attribute]
                else:
                    card_value += float(card[attribute]) * coefficients['minions'][attribute]
            card_value += coefficients['intrinsic value']
            hand_value += card_value
        elif card['type'] == 'SPELL':
            hand_value += float(card['crystals_cost'])
        elif card['type'] == 'WEAPON':
            hand_value += float(card['attack']) * coefficients['minions']['attack'] + float(card['durability']) * \
                                                                                      coefficients[
                                                                                          'weapon_durability'] + float(
                card['forgetful']) * coefficients['minions']['forgetful'] + float(
                card['freeze']) * coefficients['minions']['freezing']
    return hand_value


def calculate_real_hand_value(hand):
    value = 0
    for card in hand:
        value += card['crystals_cost']
    return value


def estimate_board_value(board):
    board_value = 0
    for card in board:
        card_value = 0
        for attribute in coefficients['minions']:
            if attribute == 'hp':
                card_value += card['hp_current'] * coefficients['minions'][attribute]
            elif attribute == 'charge':
                card_value += float(card[attribute]) * float(card['attack']) * coefficients['minions'][attribute]
            else:
                card_value += float(card[attribute]) * coefficients['minions'][attribute]
        if bool(card['can_attack']):
            card_value += coefficients['can_attack']
        if bool(card['frozen']):
            card_value += coefficients['frozen']
        board_value += card_value
    return board_value


def get_min_manacost(hand):
    min = 100
    for card in hand:
        if card['crystals_cost'] < min:
            min = card['crystals_cost']
    return min


def get_max_manacost(hand):
    max = 0
    for card in hand:
        if card['crystals_cost'] > max:
            max = card['crystals_cost']
    return max


def get_average_manacost(hand):
    if len(hand) == 0:
        return 0.0
    sum = 0.0
    for card in hand:
        sum += float(card['crystals_cost'])
    return sum / float(len(hand))


def get_std_manacost(hand):
    if len(hand) == 0:
        return 0.0
    average = get_average_manacost(hand)
    std = 0.0
    for card in hand:
        std += math.pow(float(card['crystals_cost']) - average, 2)
    std /= float(len(hand))
    std = math.sqrt(std)
    return std


def get_cards(hand):
    cards = []
    for card in hand:
        cards.append(card['crystals_cost'])
    while len(cards) < 10:
        cards.append(0)
    return cards


def create_features(data, mode = "gz", set = "train"):
    X = []
    for line in data:
        feature = []
        cards = get_cards(line['player']['hand'])
        estimated_hand_strength = estimate_hand_value(line['player']['hand'])
        real_hand_strength = calculate_real_hand_value(line['player']['hand'])
        difference = estimated_hand_strength - real_hand_strength
        player_board_strength = estimate_board_value(line['player']['played_cards'])
        enemy_board_strength = estimate_board_value(line['opponent']['played_cards'])
        player_health = int(line['player']['hero']['armor']) + int(line['player']['hero']['hp'])
        enemy_health = int(line['opponent']['hero']['armor']) + int(line['opponent']['hero']['hp'])
        player_hero_power_used = int(line['player']['hero']['special_skill_used'])
        enemy_hero_power_used = int(line['opponent']['hero']['special_skill_used'])
        player_crystals_all = int(line['player']['stats']['crystals_all'])
        player_crystals_current = int(line['player']['stats']['crystals_current'])
        enemy_crystals_all = int(line['opponent']['stats']['crystals_all'])
        enemy_crystals_current = int(line['opponent']['stats']['crystals_current'])
        player_deck_count = int(line['player']['stats']['deck_count'])
        enemy_deck_count = int(line['opponent']['stats']['deck_count'])
        player_fatique_damage = int(line['player']['stats']['fatigue_damage'])
        enemy_fatique_damage = int(line['opponent']['stats']['fatigue_damage'])
        turn = int(line['turn'])
        min_card_manacost = int(get_min_manacost(line['player']['hand']))
        max_card_manacost = int(get_max_manacost(line['player']['hand']))
        average_manacost = float(get_average_manacost(line['player']['hand']))
        std_manacost = float(get_std_manacost(line['player']['hand']))
        card_advantage = int(line['opponent']['stats']['deck_count'] - line['player']['stats']['deck_count'])
        player_played_minions_count = int(line['player']['stats']['played_minions_count'])
        enemy_played_minions_count = int(line['opponent']['stats']['played_minions_count'])
        if set == "train":
            decision = int(line['decision'])
        for card in cards:
            feature.append(card)
        feature.append(estimated_hand_strength)
        feature.append(real_hand_strength)
        feature.append(difference)
        feature.append(player_board_strength)
        feature.append(enemy_board_strength)
        feature.append(player_health)
        feature.append(enemy_health)
        feature.append(player_hero_power_used)
        feature.append(enemy_hero_power_used)
        feature.append(player_crystals_all)
        feature.append(player_crystals_current)
        feature.append(enemy_crystals_all)
        feature.append(enemy_crystals_current)
        feature.append(player_deck_count)
        feature.append(enemy_deck_count)
        feature.append(player_fatique_damage)
        feature.append(enemy_fatique_damage)
        feature.append(turn)
        feature.append(min_card_manacost)
        feature.append(max_card_manacost)
        feature.append(average_manacost)
        feature.append(std_manacost)
        feature.append(card_advantage)
        feature.append(player_played_minions_count)
        feature.append(enemy_played_minions_count)
        if set == "train":
            feature.append(decision)
        X.append(feature)
    X = np.array(X)
    print('List of features created')
    if mode == "list":
        return X
    elif mode == "gz":
        np.savetxt('trainingSet_100k_v3.gz', X, delimiter=',')