import numpy as np

# each state is a full game state, that is a vector of free cells
# (from which you can derive the number of moves so far)
# plus my location and opponent location

class multi_value_model():
    def __init__(self, state_cache, max_num = 14, mem_size =10000, look_ahead = 2):
        self.models = []
        self.states = []
        self.look_ahead = look_ahead
        for n in range(max_num):
            # create a neural network model for step n
            self.models.append(neural_model(num))
            # create state storage for step n & store pre-generated states
            self.states.append(memory(mem_size))
            self.states[-1].add(state_cache[n])

    def get_model(self, ind):
        if ind >= len(self.models):
            return (NaiveHeuristic(), float('inf'))
        else:
            return (self.models[ind], self.look_ahead)

    def generate_outputs(self, ind):
        heuristic_model, depth = self.get_model(ind + self.look_ahead)
        value_gen = agent('alphabeta', heuristic=heuristic_model, depth=depth)
        y = []
        for x in self.states[ind]:
            y.append(value_gen(x))
            return y

    def calibrate_all_models(self):
        for ind, model in reversed(enumerate(self.models)):
            x = np.array(self.states(ind))
            y = self.generate_outputs(ind)
            model.fit(x, y,...)

    def run_games(self, num_games = 1000, opponent = that_other_guy):
        for ind, model in enumerate(self.models):
            # simulate game, to begin with against the ab_improved
            pass


