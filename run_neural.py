from tournament import tournament
from reporting import Reporting
import importlib
from collections import namedtuple
from game_agent_comp import CustomPlayerComp
from value_functions import improved_score_fast_x2,improved_score_fast,\
    improved_score_fast_x3, partition_score_x2, softmax, to_index
from sample_players import null_score
from policy import SimplePolicy
import numpy as np
from neural import NNValueFunction, SelectionValueFunction, SingleValueFunction
import numpy as np
from copy import copy
from sklearn.model_selection import train_test_split
import math
import random

import pickle, glob

def nice_allscores(x, use_softmax = False):
    scores = [score for score, cell in x]
    inds = [to_index(cell) for score, cell in x]
    if use_softmax:
        return softmax(scores), inds
    else:
        max_score = max(scores)
        new_scores = np.array([1 if score == max_score else 0 for score in scores ])
        return new_scores, inds

def get_simulation_data(files):
    depths = {}
    for file in files:  # [:3]:
        try:
            with open(file, 'rb') as handle:
                old_depths = pickle.load(handle)
                # print(old_depths)
        except:
            old_depths = {}

        # print(old_depths)
        for d in old_depths:
            for p, v in d.items():
                if p not in depths:
                    depths[p] = v
                else:
                    depths[p] = depths[p] + v
    return depths


def run_batch(moves, val, train_final_scores):
    difflist = []
    train_diff = np.zeros(len(moves))
    train_base = copy(train_diff)
    dcoeff = np.zeros(val.coeff_len)
    for m, move in enumerate(moves):
        if train_final_scores:
            thisvalue = val.eval(input_vec=move['game'], pos=move['pos'], mask=move['game'])
            diff = np.array([move['score'] - thisvalue])
            train_base[m] = move['score']
            train_diff[m] = math.fabs(diff[0])
        else:
            scores, inds = nice_allscores(move['allscores'])
            thisvalue = val.eval(input_vec=move['game'], pos=move['pos'], indices=inds, mask=move['game'])
            diff = scores - thisvalue
            train_base[m] = np.linalg.norm(scores - scores.mean())
            train_diff[m] = np.linalg.norm(diff)
        difflist.append(diff)
        gr = val.nn.grad()
        delta = gr.dot(diff).transpose()
        # print(diff,dcoeff,delta)
        try:
            dcoeff += delta
        except:
            dcoeff += delta[0]
    return (dcoeff, train_base, train_diff, difflist)

def extract_moves(games):
    moves = []
    final_moves = []
    for game in games:
        for m, move in enumerate(game):
            if move['score'] != float('inf') and move['score'] != float('-inf'):
                moves.append(move)
            else:
                from copy import copy
                move_ = copy(move)
                if move['score'] == float('inf'):
                    move_['score'] = 1
                else:
                    move_['score'] = -1
                final_moves.append(move_)
    return moves, final_moves

def run_calibration(s = 0, train_final_scores = True, file_pattern = None):
    if file_pattern is None:
        if train_final_scores:
            file_pattern = 'data/ID_x2_1000ms/result_ID*.pickle'
            keyname = 'improved, two steps exact'
        else:
            file_pattern = 'data/result?.pickle'
            keyname = 'improved, two steps exact, with reporting'

    files = glob.glob(file_pattern)
    print(files)
    depths = get_simulation_data(files)
    print(depths.keys())

    if not train_final_scores:
        example = depths[keyname][0]
        move = example[1]
        print(move['allscores'])
        print(nice_allscores(move['allscores']))

    # for player, games in depths.items():
    games = depths[keyname]
    games_train, games_test = train_test_split(games, test_size=0.1)


    moves_train, final_moves_train = extract_moves(games_train)
    moves_test, final_moves_test = extract_moves(games_test)
    print(len(moves_train), len(final_moves_train))
    #print(nice_allscores(moves_train[0]['allscores']))

    if train_final_scores:
        train_moves, test_moves = final_moves_train, final_moves_test
        sizes = [[3,3], [3,3,3], [3,5,5,3], [4,5,4]]
        val = SingleValueFunction(sizes[s])
        coeff = np.random.normal(size=val.coeff_len) * 0.5
        grad_mult = 0.00001
    else:
        train_moves, test_moves = moves_train, moves_test
        sizes = [[2, 2, 2], [2,2,2,2], [2,2,2,2,2], [3,3,3]]
        val = SelectionValueFunction(sizes[s])
        coeff = np.random.normal(size=val.coeff_len) * 0.1
        grad_mult = 0.01


    n = 0
    while 53 * n < len(coeff):
        coeff[53 * n:(53 * n + 10)] = 0
        n += 1

    val.set_coeff(coeff)

    n = 0

    test_diff = np.zeros(len(test_moves))
    test_base = copy(test_diff)

    timed = False
    epoch = 0
    trainerr = []
    testerr = []
    while True:
        print('entering calibration next epoch...')

        random.shuffle(train_moves)
        # just run one batch
        batch_size = 100
        for m in range(int(len(train_moves)/batch_size)):
            moves = train_moves[m*batch_size:(m+1)*batch_size]
            dcoeff, train_base, train_diff, difflist = run_batch(moves,val,train_final_scores)
            # print(np.linalg.norm(dcoeff), np.linalg.norm(val.nn.get_coeff()))
            # dnorm = np.linalg.norm(dcoeff)
            # cnorm = np.linalg.norm(val.nn.get_coeff())
            # dcoeff = math.sqrt(cnorm/dnorm)*dcoeff
            # if m%50 == 0 and m>0:
            #print(np.linalg.norm(dcoeff*grad_mult), np.linalg.norm(val.nn.get_coeff()))
            #print('test: ',np.linalg.norm(train_diff)/np.linalg.norm(train_base))
            coeff = val.nn.get_coeff() + dcoeff*grad_mult
            # print(coeff.shape)
            val.nn.set_coeff(coeff)

            random.shuffle(test_moves)
            for m, move in enumerate(test_moves[:batch_size]): # TODO: factor this code into run_batch
                if train_final_scores:
                    thisvalue = val.eval(input_vec=move['game'], pos=move['pos'], mask=move['game'])
                    diff = move['score'] - thisvalue
                    test_base[m] = move['score']
                else:
                    scores, inds = nice_allscores(move['allscores'])
                    thisvalue = val.eval(input_vec=move['game'], pos=move['pos'], indices=inds, mask=move['game'])
                    diff = scores - thisvalue
                    test_base[m] = np.linalg.norm(scores - scores.mean())

                test_diff[m] = np.linalg.norm(diff)

            #print(np.linalg.norm(grad_mult*dcoeff), np.linalg.norm(val.nn.get_coeff()))
            trainerr.append(1 - (np.linalg.norm(train_diff) / np.linalg.norm(train_base)) ** 2)
            testerr.append(1 - (np.linalg.norm(test_diff) / np.linalg.norm(test_base)) ** 2)
            #print('train: ', trainerr[-1])
            #print('test: ', testerr[-1])
        epoch += 1
        if train_final_scores:
            mystr = 'final_'
        else:
            mystr = ''
        print('dumping data...')
        with open('data/ID_x2_1000ms/calibrated_new2_' + mystr + str(s) + '_epoch_' + str(epoch) + '.pickle', 'wb') as handle:
            pickle.dump({'function': val, 'trainerr' : trainerr, 'testerr': testerr}, handle)