import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from neural.neural_ import get_legal_moves, to_index
from constants import BOARD_SIZE

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