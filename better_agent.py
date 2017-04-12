import numpy as np
from collections import namedtuple
# define mapping from coeffs to
def to_index(pair, board_size = 7):
    return pair[0]+ board_size*pair[1]

def to_pair(index, board_size = 7): # correct?
    return (index%board_size, int(index/board_size))

def generate_all_moves_by_index():
    from sample_players import RandomPlayer
    from isolation import Board

    move_dict = {}
    player1 = RandomPlayer()
    player2 = RandomPlayer()
    game = Board(player1, player2)
    all_moves = game.get_legal_moves(player1)
    print(len(all_moves))
    for move in all_moves:
        new_game = game.forecast_move(move)
        move_dict[to_index(move)] =\
            [to_index(x) for x in new_game.get_legal_moves(player1)]
        #print(len(move_dict[move]))
    return move_dict


def rotation_matrices():
    matrix_array = []
    for i1 in [-1,1]:
        for i2 in [-1,1]:
            matrix_array.append(np.array([[0, i1], [i2, 0]]))
            matrix_array.append(np.array([[i1, 0], [0, i2]]))

    matrix_array.pop(-1)
    return matrix_array

def rotate_index(ind, mat):
    pair = to_pair(ind)
    point = np.array(pair) - 3.5
    r_point = (mat.dot(point) + 3.6).astype(int)
    return to_index(r_point)

def correct_octile(index):
    pair = to_pair(index)
    return pair[0] <=3 and pair[1] <=3 and pair[0] <= pair[1]

def move_convolution_indices():
    move_dict = generate_all_moves_by_index()
    cell_index_dict = {}
    coeff_index_dict = {}
    # first pass: for the preferred octile fields, populate coeff_index with sequential coeffs
    # TODO: add self to list of neighbors
    # TODO: additional symmetry in the correct octile coeffs
    # TODO: make sure own coeffs are 0:48
    n = 0
    for index, value in move_dict.items():
        if correct_octile(index):
            cell_index_dict[index] = move_dict[index]
            coeff_index_dict[index] = []
            for i in range(len(value)):
                coeff_index_dict[index].append(n)
                n += 1

    # second pass: for all other moves x, find F that moves x into the right octile,
    # assign to x the same coeff_indices as to F(x), and the same cell_index as F_inv(cell_index(F(x))
    mat_list = rotation_matrices()
    if True:
        for index, value in move_dict.items():
            if not correct_octile(index):
                for mat in mat_list:
                    rotated_index = rotate_index(index, mat)
                    if correct_octile(rotated_index):
                        break

                inv_mat = np.linalg.inv(mat)
                coeff_index_dict[index] = []
                cell_index_dict[index] = []
                for i in range(len(value)):
                    coeff_index_dict[index].append(coeff_index_dict[rotated_index][i])
                    cell_index_dict[index].append(rotate_index(cell_index_dict[rotated_index][i], inv_mat))
        # now join them up
    joint_indices = {}
    Point = namedtuple('Point',['field','coeff'])

    for index, value in cell_index_dict.items():
            joint_indices[index] = []
            for i in range(len(value)):
                joint_indices[index].append((cell_index_dict[index][i],coeff_index_dict[index][i]))

    return joint_indices, n

# TODO: calculate once (or load from json), get from singleton
class MoveConvolutionIndices:
    def __init__(self):
        self.joint_indices, self.coeff_len = move_convolution_indices()

    def __call__(self):
        return self.joint_indices, self.coeff_len

def move_convolution(move_vec, coeff_vec, joint_indices,
                     mask = None, result_vector = np.array([0]*49)):
    for index, indlist in joint_indices:
        result_vector[index] = 0
        # if some of the fields have to be 0 regardless, no point in computing them
        if mask is None or mask[index] > 0:
            for i in indlist:
                result_vector[index] += move_vec[i[0]] * coeff_vec[i[1]]

    return result_vector


def move_convolution_grad(move_vec, coeff_vec, joint_indices,
                          mask = None, result_grad = None):
    if result_grad is None:
        result_vector= np.zeros([len(move_vec), len(coeff_vec)])

    for index, indlist in joint_indices:
        if mask is None or mask[index] > 0:
            for i in indlist:
                result_grad[index, i[1]] = move_vec[i[0]]

    return result_grad


class ConvolutionUnit:
    def __init__(self, coeffs = None, output_vec = None, input_vec = None, mask = None):

        self.joint_indices, self.coeff_len = MoveConvolutionIndices()()

        if coeffs is None:
            self.coeffs = np.zeros([self.coeff_len])
        else:
            self.coeffs = coeffs

        if input_vec is None:
            self.input_vec = np.zeros([7*7])# TODO: get size from static config singleton
        else:
            self.input_vec = input_vec

        if mask is None:
            self.mask = np.zeros([7*7])# TODO: get size from static config singleton
        else:
            self.mask = input_vec

        if output_vec is None:
            self.output_vec = np.zeros([7*7])
        else:
            self.output_vec = output_vec

    def __call__(self, input_vec = None):
        if input_vec is None:
            input_vec = self.input_vec

        return move_convolution(input_vec, self.coeffs, self.joint_indices, self.mask, self.output_vec)

    def grad(self, input_vec=None):
        if input_vec is None:
            input_vec = self.input_vec

        return move_convolution(input_vec, self.coeffs, self.joint_indices, self.mask, self.output_vec)

class ConvolutionStage:
    '''
    A stage converting multiple channels to more multiple channels
    '''
    def __init__(self, dim1, dim2, coeffs=None, output_vec=None, input_vec=None, mask=None):
        self.units = []
        self.coeff_len = ConvolutionUnit().coeff_len # number of coefficients per conv unit
        for d1 in range(dim1):
            self.units.append([])
            for d2 in range(dim2):
                self.units[-1].append(ConvolutionUnit(
                    coeffs[(d1*dim2 + d2)*self.coeff_len:(d1*dim2 + d2 + 1)*self.coeff_len],
                    input_vec[d1*49:(d1+1)*49],
                    output_vec[d2*49:(d2+1)*49],
                    mask))


class ConvolutionNetwork:
    def __init__(self, dims):
        self.unit_coeff_len = ConvolutionUnit().coeff_len  # number of coefficients per conv unit
        self.total_coeff_len = 0
        dim_pairs = []
        for i in range(len(dims)-1):
            this_dim_pair = [dims[i], dims[i+1]]
            dim_pairs.append(this_dim_pair)
            self.total_coeff_len += this_dim_pair[0]*this_dim_pair[1]*self.unit_coeff_len

        # spawn correct number of convolution units

    def coeffs_len(self):
        return 0 #number_of_coeffs

    def get_coeff_vector(self):
        pass

    def set_coeff_vector(self):
        pass

    def __call__(self, input_vector):
        # do convolution on all fronts
        pass


# so, the convolution network is my value function
conv_network_instance = ConvolutionNetwork([2, 1])

class NNValueFunction():
    def __init__(self,dims, coeffs):
        self.nn = ConvolutionNetwork(dims)
        self.nn.set_coeff_vector(coeffs)

    def __call__(self, game, player):
        pass
# in a while-loop later, now just one pass to make sure it works
'''
# first, run vanilla minimax with ID and 2x_exact heuristic against some opponents, 
# for each get_move storing the tuple of scores for the level below top
# sample these, (game, top-level scores) and use to train a policy NN

# also store the game states where the heuristic function was called, along with heuristic
# function output

# now do preliminary training of policy function on the first set

# then do prelim training of value nn function on second set

list_of_opponents = [ID_Improved]
new_coeff_vector = ...
if True:
    my_agent = game_agent(NNValueFunction(dims, new_coeff_vector))
    results = run_tournament(my_agent, list_of_opponents, num_games = 100, num_cores = 1)
    training_set = extract_training_set(results)
    coeff_vector = train_network(my_agent.score, training_set, epochs = 10) # also displays training graphs in and out of sample
    list_of_opponents.append(my_agent)
    print(win_ratio(results))
'''

if __name__ == "__main__":
    move_convolution_indices()




