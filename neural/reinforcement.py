from collections import deque
from math import sqrt
import numpy as np
from data_utils import prepare_data_for_model
from tournament import tournament, Agent, CustomPlayerComp, improved_score_fast_x2
from data_utils import get_depths

class Memory:
    def __init__(self, max_size=1000, sampling_ratio=1.01):
        self.buffer = deque(maxlen=max_size)
        self.ratio = sampling_ratio

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        losses = np.array([sqrt(loss) for b, loss in self.buffer])
        med = np.median(losses)
        losses[losses < med / self.ratio] = med / self.ratio
        losses[losses > med * self.ratio] = med * self.ratio
        p = losses / sum(losses)

        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False,
                               p=p)
        return idx, [self.buffer[i] for i in idx]

    def replace(self, experience, index):
        self.buffer[index] = experience


def generate_target(states, deep_Q_model, alpha=0.5, discount_factor=0.99):
    board_full, player_pos, legal_moves, next_move, G = prepare_data_for_model(states, 'G')
    # get list of next states for approx
    # for states where a valid move was made, get the value of the resulting state
    next_states = [state['next_state'] for state in states]
    valid_next_ind = [n for n, state in enumerate(next_states) if state is not None]
    valid_next_states = [state for state in next_states if state is not None]

    board_full_n, player_pos_n, legal_moves_n, _, _ = prepare_data_for_model(valid_next_states, None)
    next_Q_values = deep_Q_model.predict([board_full_n, player_pos_n, legal_moves_n])
    # one minus because next turn the adversary chooses the best for them, which is the worst for us
    # incidentally, if the adversary has no valid moves, the Q model will return all 0s, which is correct
    best_next_Q = 1 - next_Q_values.max(axis=1)

    # if the state didn't include a valid next state, means we lost that time, so state value is 0
    full_next_Q = np.zeros(len(states))
    full_next_Q[valid_next_ind] = best_next_Q
    # TODO: handle end state rewards better in reporting
    target_from_Q = 0.5 + discount_factor * (full_next_Q - 0.5)
    target_from_G = G.T[0]
    target = alpha * target_from_Q + (1 - alpha) * target_from_G
    return board_full, player_pos, legal_moves, next_move, target #, target_from_Q, target_from_G.T[0]

def run_tournament(trainee_, num_rounds = 10, time_limit=float('inf'), discount_factor=0.99 ):
    result, win_ratio = tournament(num_rounds, test_agents = [trainee_], time_limit = time_limit)
    #print(result)
    nice_data = get_depths(result, [trainee_], lambda x:x, discount_factor)
    #print(nice_data)
    # TODO: nicer handling of final states, so wins/losses also propagate from final values via Q, not just via G
    states = [state for game in nice_data[trainee_.name] for state in game]
    print('imported',len(states), 'states')
    return states, result, win_ratio

