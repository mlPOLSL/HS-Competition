import json
import numpy as np

j = {}
trainingpaths = ["C:\\Users\\user\PycharmProjects\Hearthstone\\trainingData_JSON\\trainingData_JSON_chunk1.json",
                 "C:\\Users\\user\\PycharmProjects\Hearthstone\\trainingData_JSON\\trainingData_JSON_chunk2.json",
                 "C:\\Users\\user\\PycharmProjects\Hearthstone\\trainingData_JSON\\trainingData_JSON_chunk3.json",
                 "C:\\Users\\user\PycharmProjects\Hearthstone\\trainingData_JSON\\trainingData_JSON_chunk4.json"]
testpaths = ["C:\\Users\\user\PycharmProjects\Hearthstone\\testData_JSON\\testData_JSON_chunk8.json",
             "C:\\Users\\user\\PycharmProjects\Hearthstone\\testData_JSON\\testData_JSON_chunk9.json",
             "C:\\Users\\user\\PycharmProjects\Hearthstone\\testData_JSON\\testData_JSON_chunk10.json"]
deprecated_testpaths = ["C:\\Users\\user\PycharmProjects\Hearthstone\deprecated_testData_JSON\\testData_JSON_chunk5.json",
                        "C:\\Users\\user\PycharmProjects\Hearthstone\deprecated_testData_JSON\\testData_JSON_chunk6.json",
                        "C:\\Users\\user\PycharmProjects\Hearthstone\deprecated_testData_JSON\\testData_JSON_chunk7.json"]

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
            hand_value += float(card['attack']) * coefficients['minions']['attack'] + float(card['durability']) * coefficients[
                'weapon_durability'] + float(card['forgetful']) * coefficients['minions']['forgetful'] + float(
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


X = []
x = []
for i, path in enumerate(deprecated_testpaths, 1):
    with open(path) as file:
        print('opening another file')
        X.clear()
        for line in file:
            j = json.loads(line)
            x.clear()
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
            player_spell_dmg_bonus = int(j['player']['stats']['spell_dmg_bonus'])
            enemy_spell_dmg_bonus = int(j['opponent']['stats']['spell_dmg_bonus'])
            turn = int(j['turn'])
            # decision = int(j['decision'])
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
            x.append(player_spell_dmg_bonus)
            x.append(enemy_spell_dmg_bonus)
            x.append(turn)
            # x.append(decision)
            X.append(x.copy())
        X_ = np.array(X.copy())
        np.savetxt('deprecated_testSet' + str(i) + '.gz', X_, delimiter=',')

