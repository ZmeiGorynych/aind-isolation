import numpy as np

from constants import BOARD_SIZE
from data_utils import prepare_data_for_model, apply_move
from neural.neural_ import to_pair, get_legal_moves
from keras.activations import softmax

# in the below, game is a dict with the fields
# 'pos' is [pos of player about to move, other_pos]
# 'game' is a vector of 0s for used fields, 1s for available fields

# def get_best_move_from_model(game, model):
#     moves = get_legal_moves(game)
#     if len(moves):
#         tmp = [apply_move(game, move) for move in moves]
#         #print(tmp)
#         board, pos, _ = prepare_data_for_model(tmp,None)
#         valuations = 1-model.predict([pos, board])
#         best_ind = np.argmax(valuations)
#         #print(best_ind)
#         return moves[best_ind], valuations[best_ind]
#     else:
#         return (0,float('-inf'))
def get_best_move_from_model(game, model, temperature = 1.0):
    board, pos, legal_moves, _, _ = prepare_data_for_model([game], None)
    move_probs = model.predict([board, pos, legal_moves])[0].T[0]
    if True:
        best_move = np.argmax(move_probs)
        #print(move_probs)
    else:
        leg_moves = get_legal_moves(game)
        if not len(leg_moves): # if no legal moves left
            return (-1,-1)
        move_probs_legal = move_probs[leg_moves]
        tmp = np.exp(move_probs_legal)
        p = tmp/sum(tmp)
        best_move = np.random.choice(leg_moves, p=p)
    return best_move, move_probs[best_move]




class NeuralAgent:
    def __init__(self, model, name = None, temperature = 1.0):
        self.model = model
        self.name = name
        self.temperature = temperature

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
        move, score = get_best_move_from_model(info, self.model, self.temperature)
        #print('yippee',pos, get_legal_moves(info), move, score)
        info['move'] = move
        info['n_score'] = score
        return  (to_pair(move), info)

if __name__ == '__main__': # sample testing code
    board = np.ones(BOARD_SIZE)
    my_pos = None
    other_pos = None
    game = {'pos': np.array([my_pos, other_pos]), 'game': board}
    game1 = apply_move(game, 0)
    game2 = apply_move(game1, 1)
    game3 = apply_move(game2, 15)