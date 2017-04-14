import numpy as np
from collections import namedtuple
# define mapping from coeffs to
def to_index(pair, board_size = 7):
    return pair[0]+ board_size*pair[1]

def to_pair(index, board_size = 7): # correct?
    return (index%board_size, int(index/board_size))

def square_print(vec):
    for i in range(7):
        print(vec[i*7:(i+1)*7])

def generate_all_moves_by_index():
    from sample_players import RandomPlayer
    from isolation import Board

    move_dict = {}
    player1 = RandomPlayer()
    player2 = RandomPlayer()
    game = Board(player1, player2)
    all_moves = game.get_legal_moves(player1)
    #print(len(all_moves))
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
    point = np.array(pair) - 3
    r_point = (mat.dot(point) + 3.1).astype(int)
    return to_index(r_point)


def correct_octile(index):
    pair = to_pair(index)
    return pair[1] <=3 and pair[0] <= pair[1]

def move_convolution_indices():
    move_dict = generate_all_moves_by_index()
    cell_index_dict = {}
    coeff_index_dict = {}
    # first pass: for the preferred octile fields, populate coeff_index with sequential coeffs
    # TODO: additional symmetry in the correct octile coeffs
    #make sure offsets are 0:10, own coeffs 10:20
    own_n = 0
    n = 20
    for index, value in move_dict.items():
        if correct_octile(index):
            cell_index_dict[index] = [-1, index] + move_dict[index] #add self to list of neighbors
            coeff_index_dict[index] = [own_n, own_n + 10] # make sure offsets are 0:10, own coeffs 10:20
            own_n += 1
            # additional symmetry in the correct octile coeffs
            pair = to_pair(index)
            if pair[0]==pair[1]:
                mat = np.array([[0,1],[1,0]])
            elif pair[1] == 3:
                mat = np.array([[1,0],[0,-1]])
            else:
                mat = None

            if mat is None:
                for i in range(len(value)):
                    coeff_index_dict[index].append(n)
                    n += 1
            else:
                moves_so_far = set()
                for ind in move_dict[index]:
                    twin = rotate_index(ind, mat)
                    if twin not in moves_so_far:
                        coeff_index_dict[index].append(n)
                        moves_so_far.add(ind)
                        n += 1
                    else:
                        twin_coeff = [c for ci, c in enumerate(coeff_index_dict[index][2:]) if move_dict[index][ci] == twin][0]
                        coeff_index_dict[index].append(twin_coeff)

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

                coeff_index_dict[index] = coeff_index_dict[rotated_index]
                inv_mat = np.linalg.inv(mat)
                cell_index_dict[index] = [-1, index]
                for cind in cell_index_dict[rotated_index][2:]:
                    cell_index_dict[index].append(rotate_index(cind, inv_mat))
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
    for index, indlist in joint_indices.items():
        # first coeff is offset
        result_vector[index] = coeff_vec[indlist[0][1]]
        # if some of the fields have to be 0 regardless, no point in computing them
        if mask is None or mask[index] > 0:
            for i in indlist[1:]:
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
    def __init__(self):

        self.joint_indices, self.coeff_len = MoveConvolutionIndices()()
        self.input_len = 7*7
        self.output_len = 7*7
        self.output_vec = np.zeros(self.output_len)

    def set_coeff(self, coeff):
        self.coeff = coeff

    def __call__(self, input_vec, mask = None):
        self.input_vec = input_vec
        return move_convolution(input_vec, self.coeff, self.joint_indices, mask, self.output_vec)

    def grad(self, input_vec, mask = None):
        return move_convolution_grad(input_vec, self.coeff, self.joint_indices, mask, self.output_vec)

class ConvolutionStage:
    '''
    A stage converting multiple channels to more multiple channels
    '''
    def __init__(self, dim1, dim2):
        self.units = []
        self.dim1 = dim1
        self.dim2 = dim2
        self.coeff_len = ConvolutionUnit().coeff_len * dim1 * dim2
        for d1 in range(dim1):
            self.units.append([])
            for d2 in range(dim2):
                self.units[-1].append(ConvolutionUnit())
        self.input_len = dim1 * self.units[0][0].input_len
        self.output_len = dim2 * self.units[0][0].output_len
        self.output_vec = np.zeros(self.output_len)

    def set_coeff(self, coeff):
        self.coeff = coeff
        startind = 0
        for ul in self.units:
            for u in ul:
                finalind = startind + u.coeff_len
                u.set_coeff(coeff[startind:finalind])
                startind = finalind

    def get_coeff(self):
        return self.coeff

    def __call__(self, input_vec, mask = None):
        self.input_vec = input_vec
        self.output_vec[:] = 0
        uolen = self.units[0][0].output_len
        uilen = self.units[0][0].input_len
        for d1, ul in enumerate(self.units):
            for d2, u in enumerate(ul):
                self.output_vec[d2*uolen: (d2+1)*uolen] += u(input_vec[d1*uilen:(d1+1)*uilen], mask)

        return self.output_vec

class ConvolutionNetwork:
    def __init__(self, dims):
        self.coeff_len = 0
        self.stages = []

        # spawn correct number of convolution units
        for i in range(len(dims)-1):
            self.stages.append(ConvolutionStage(dims[i], dims[i+1]))

        self.input_len = self.stages[0].input_len
        self.output_len = self.stages[-1].output_len

        self.coeff_len = 0
        for s in self.stages:
            self.coeff_len += s.coeff_len

    def get_coeff(self):
        return self.coeff

    def set_coeff(self, coeff):
        self.coeff = coeff
        # split it into bits and assign those to the units
        startind = 0
        for s in self.stages:
            finalind = startind + s.coeff_len
            s.set_coeff(coeff[startind:finalind])
            startind = finalind

    def __call__(self, input_vector, mask = None):
        self.input_vector = input_vector

        for s in self.stages:
            input_vector = s(input_vector, mask)

        return input_vector
# so, the convolution network is my value function

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




