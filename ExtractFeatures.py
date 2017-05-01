import json
import numpy as np
import math
from paths import original_deprecated_testpaths, original_test_paths, original_training_paths

j = {}

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
    'can_attack': 1.5,
    'frozen': -1.5
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


X = []
x = []
for i, path in enumerate(original_test_paths, 1):
    with open(path) as file:
        print('opening another file')
        X.clear()
        for line in file:
            j = json.loads(line)
            x.clear()
            cards = get_cards(j['player']['hand'])
            hand_strength = estimate_hand_value(j['player']['hand'])
            player_board_strength = estimate_board_value(j['player']['played_cards'])
            enemy_board_strength = estimate_board_value(j['opponent']['played_cards'])
            player_health = int(j['player']['hero']['armor']) + int(j['player']['hero']['hp'])
            enemy_health = int(j['opponent']['hero']['armor']) + int(j['opponent']['hero']['hp'])
            player_hero_power_used = int(j['player']['hero']['special_skill_used'])
            enemy_hero_power_used = int(j['opponent']['hero']['special_skill_used'])
            player_crystals_all = int(j['player']['stats']['crystals_all'])
            player_crystals_current = int(j['player']['stats']['crystals_current'])
            enemy_crystals_all = int(j['opponent']['stats']['crystals_all'])
            enemy_crystals_current = int(j['opponent']['stats']['crystals_current'])
            player_deck_count = int(j['player']['stats']['deck_count'])
            enemy_deck_count = int(j['opponent']['stats']['deck_count'])
            player_fatique_damage = int(j['player']['stats']['fatigue_damage'])
            enemy_fatique_damage = int(j['opponent']['stats']['fatigue_damage'])
            turn = int(j['turn'])
            min_card_manacost = int(get_min_manacost(j['player']['hand']))
            max_card_manacost = int(get_max_manacost(j['player']['hand']))
            average_manacost = float(get_average_manacost(j['player']['hand']))
            std_manacost = float(get_std_manacost(j['player']['hand']))
            card_advantage = int(j['opponent']['stats']['deck_count'] - j['player']['stats']['deck_count'])
            player_played_minions_count = int(j['player']['stats']['played_minions_count'])
            enemy_played_minions_count = int(j['opponent']['stats']['played_minions_count'])
            # decision = int(j['decision'])
            for card in cards:
                x.append(card)
            x.append(hand_strength)
            x.append(player_board_strength)
            x.append(enemy_board_strength)
            x.append(player_health)
            x.append(enemy_health)
            x.append(player_hero_power_used)
            x.append(enemy_hero_power_used)
            x.append(player_crystals_all)
            x.append(player_crystals_current)
            x.append(enemy_crystals_all)
            x.append(enemy_crystals_current)
            x.append(player_deck_count)
            x.append(enemy_deck_count)
            x.append(player_fatique_damage)
            x.append(enemy_fatique_damage)
            x.append(turn)
            x.append(min_card_manacost)
            x.append(max_card_manacost)
            x.append(average_manacost)
            x.append(std_manacost)
            x.append(card_advantage)
            x.append(player_played_minions_count)
            x.append(enemy_played_minions_count)
            # x.append(decision)
            X.append(x.copy())
        X_ = np.array(X.copy())
        np.savetxt('testSet' + str(i) + '_v3.gz', X_, delimiter=',')
