import copy
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from neural.neural_ import get_legal_moves, to_index
from constants import BOARD_SIZE
from value_functions import game_vector, to_index


def load_simulation_data(files):
    depths = {}
    for file in files:#[:3]:
        try:
            with open(file, 'rb') as handle:
                old_depths = pickle.load(handle)
                #print(old_depths)
        except:
            old_depths={}

        #print(old_depths)
        for d in old_depths:
            for p, v in d.items():
                if p not in depths:
                    depths[p]=v
                else:
                    depths[p] = depths[p]+v
    return depths

def transpose_list_of_lists(lol):
    
    transp = []

    for game in lol:
        for m, move_depth in enumerate(game):
            if len(transp) <= m:
                transp.append([])
            transp[m].append(move_depth)
    return transp


def prepare_data_for_model(states, score_name = 'simple_score'):
    if score_name is not None:
        y = np.array([state[score_name] for state in states])
        y = y[:,None]
    else:
        y = None
    board = np.array([list(state['game']) for state in states])
    pos = np.array([list(state['pos']) for state in states])
    try:
        move = np.array([state['move'] for state in states])
    except:
        move = None

    #print(board.shape,pos.shape, y.shape if y is not None else None)
    encoder = OneHotEncoder(BOARD_SIZE)
    pos_oh = encoder.fit_transform(pos).toarray()

    # now make sure they're the right shape
    player_pos_one_hot_value = np.array( np.concatenate( [pos_oh[:,:BOARD_SIZE,None],pos_oh[:,BOARD_SIZE:,None]],2))
    board_full = np.array(np.reshape(board, [board.shape[0],BOARD_SIZE,1]))
    # produce dummies for all possible next moves and for the actual next move
    next_move = np.zeros([len(states), BOARD_SIZE, 1])
    valid_moves = np.zeros([len(states), BOARD_SIZE, 1])

    for s, state in enumerate(states):
        if move is not None:
            next_move[s, move[s], 0] = 1
        legal_inds = get_legal_moves(state)
        for i in legal_inds:
            valid_moves[s, i, 0] = 1

    valid_moves = valid_moves*board_full # can only move to an unused field
    #print(board_full.shape,player_pos_one_hot_value.shape)
    return board_full, player_pos_one_hot_value, valid_moves, next_move, y


def apply_move(game, move):
    if not move in get_legal_moves(game):
        raise ValueError('Illegal move!')
    new_board = copy.copy(game['game'])
    new_board[move] = 0
    other_pos = move
    moving_pos = game['pos'][1]
    return {'game': new_board, 'pos': np.array([moving_pos, other_pos])}


def get_depths(report, test_agents, func=lambda x: x['depth'], disc_factor = 0.99):
    # get player names:
    players = []
    for game in report:
        for move in game['moves']:
            try:
                full_player = [p for p in test_agents if p.player == move['active_player']][0]
                if full_player not in players:
                    players.append(full_player)
            except:
                pass

    #print('****', players)
    depths = {}
    out_depths = {}
    Gs = set()
    for p in players:
        depths[p] = []
        for game in report:
            winner = float(game['winner'] == p.player)
            depths[p].append([])
            for m, move_ in enumerate(game['moves']):
                try:
                    move = copy.copy(move_)
                    if move['active_player'].name == p.player.name:
                        move['winner'] = winner
                        move['game'], move['pos'] = game_vector(move['game_'],p.player)
                        try: # if the move is a pair rather than an index
                            move['move'] = to_index(move['move'])
                        except:
                            pass

                        try: #this will fail if this is the final state
                            move['next_state'] = apply_move(move,move['move'])
                        except:
                            move['next_state'] = None
                        move['game_'] = None
                        move['active_player'] = None
                        moves_left = len(game['moves']) - (m + 1)# want an exponent of 0 at the last index
                        # discounted final reward
                        move['G'] = 0.5 + (winner - 0.5)*(disc_factor**moves_left)
                        Gs.add(move['G'])
                        depths[p][-1].append(func(move))
                except:
                    pass

        clean = []
        for game in depths[p]:
            if game:
                clean.append(game)

        out_depths[p.name] = clean
    #print(list(Gs))
    return out_depths