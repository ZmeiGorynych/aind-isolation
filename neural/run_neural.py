import sys
sys.path.append('..')
import pickle
from neural.reinforcement import Memory, run_tournament
from game_agent_comp import CustomPlayerComp, improved_score_fast_x2
from tournament import Agent
from neural.keras_utils import deep_model_fun
from neural.reinforcement import generate_target
from neural.neural_agent import create_neural_agent

def model_fun():
    return deep_model_fun(num_features =8, num_res_modules = 8, drop_rate = 0.1, activation = 'sigmoid')

mem_size = 1000000
batch_size = 10000
train_batch_size = 16
num_rounds = 10
num_init_rounds = 1000
initial_sim = False
#disc_factor = 0.99
learning_rate = 0.0001

dummy_loss = 1.0 # later want to oversample high losses
memory = Memory(mem_size)

if initial_sim:
    # Run this to generate the initial simulation data to pre-fit the model
    trainee = Agent(CustomPlayerComp(score_fn=improved_score_fast_x2,
                                     name = "Trainee",
                                     method ='alphabeta',
                                     iterative = True),
                    "Trainee")

    states, result, win_ratio= run_tournament(trainee,num_init_rounds, time_limit = 150)

    for state in states:
        memory.add((state, dummy_loss))


    with open('../data/initial_run.pickle', 'wb') as handle:
        pickle.dump(states, handle)
else:
    with open('../data/initial_run.pickle', 'rb') as handle:
        states = pickle.load(handle)

i = 0


my_agent, deep_model, deep_Q_model = create_neural_agent(model_fun, name = 'Trainee')
win_ratios = []
while True:
    i += 1
    print('*** Iteration', i, '***')
    for state in states:
        memory.add((state, dummy_loss))
    print(len(memory.buffer))

    idx, batch_states = memory.sample(batch_size)
    batch_states = [b[0] for b in batch_states]

    board_full, player_pos, legal_moves, next_move, target = generate_target(batch_states, deep_Q_model, alpha=1.0,
                                                                             discount_factor=0.99)
    deep_model.fit([board_full, player_pos, legal_moves, next_move],
                   target,
                   batch_size=train_batch_size,
                   epochs=1,
                   verbose=0,
                   validation_split=0.1,
                   shuffle=True)
    deep_Q_model.save_weights('../data/deep_Q_model_weights_' + str(i) + '.h5')

    my_agent.player.temperature = 1 / i
    states, result, win_ratio = run_tournament(my_agent, num_rounds)
    win_ratios.append(win_ratio)
    with open('../data/win_ratios.pickle', 'wb') as handle:
        pickle.dump(win_ratios, handle)