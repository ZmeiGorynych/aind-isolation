from copy import copy
import numpy as np
from neural.neural_ import generate_all_moves_by_index, to_pair
from neural.data_utils import prepare_data_for_model
# in the below, game is a dict with the fields
# 'pos' is [pos of player about to move, other_pos]
# 'game' is a vector of 0s for used fields, 1s for available fields
move_dict = generate_all_moves_by_index()

def get_legal_moves(game):
    if game['pos'][0] is None:
        return [m for m in range(49) if game['game'][m] == 1]
    else:
        moves = move_dict[game['pos'][0]]
        return [m for m in moves if game['game'][m] == 1]

def apply_move(game, move):
    if not move in get_legal_moves(game):
        raise ValueError('Illegal move!')
    new_board = copy(game['game'])
    new_board[move] = 0
    other_pos = move
    moving_pos = game['pos'][1]
    return {'game': new_board, 'pos': np.array([moving_pos, other_pos])}

def get_best_move_from_model(game, model = None):
    moves = get_legal_moves(game)
    if len(moves):
        tmp = [apply_move(game, move) for move in moves]
        #print(tmp)
        board, pos, _ = prepare_data_for_model(tmp,None)
        valuations = 1 - model.predict([pos, board])
        best_ind = np.argmax(valuations)
        #print(best_ind)
        return moves[best_ind], valuations[best_ind]
    else:
        return (0,float('-inf'))

class NeuralAgent:
    def __init__(self, model):
        self.model = model

    def get_move(self, game, *args):
        try: # extact board and player position info from a Udacity game class
            board = 1-np.array(game._board_state[:-3])
            active_player = -2 if game._board_state[-3] else -1
            other_player =  -1 if game._board_state[-3] else -2
            pos = np.array([game._board_state[active_player], game._board_state[other_player]])
            info = {'game':board,'pos':pos}
        except:
            if 'game' in game and 'pos' in game: # if we're dealing with our-style dict
                info = game
            else:
                raise ValueError('game must be either a Udacity-style game or a dict with fields game and pos')
        #print('info', info)
        move, score = get_best_move_from_model(info, self.model)
        #print('yippee',pos, get_legal_moves(info), move, score)
        info['move'] = move
        info['n_score'] = score
        return  (to_pair(move), info)

if __name__ == '__main__': # sample testing code
    board = np.ones(49)
    my_pos = None
    other_pos = None
    game = {'pos': np.array([my_pos, other_pos]), 'game': board}
    game1 = apply_move(game, 0)
    game2 = apply_move(game1, 1)
    game3 = apply_move(game2, 15)