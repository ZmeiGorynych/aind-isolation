from tournament import tournament
from reporting import Reporting
import importlib
from collections import namedtuple
from game_agent_comp import CustomPlayerComp
from value_functions import improved_score_fast_x2,improved_score_fast,\
    improved_score_fast_x3, partition_score_x2, nice_allscores
from sample_players import null_score
from policy import SimplePolicy
import numpy as np

import pickle, glob

def run_calibration(s, train_final_scores = False ):
    #train_final_scores = False
    #s = 1

    files = glob.glob('data/result*.pickle')
    print(files)

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

    #with open('result.pickle', 'wb') as handle:
    #    pickle.dump(depths, handle)
    print(depths.keys())

    example = depths['improved, three steps exact'][0]
    move = example[1]
    print(move['allscores'])

    print(nice_allscores(move['allscores']))

    moves = []
    final_moves = []
    # for player, games in depths.items():
    games = depths['improved, three steps exact']
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

    depths = {}

    print(len(moves), len(final_moves))
    print(nice_allscores(moves[0]['allscores']))

    from neural import NNValueFunction, SelectionValueFunction, SingleValueFunction
    import numpy as np
    from copy import copy
    from sklearn.model_selection import train_test_split
    import math
    import random


    if train_final_scores:
        train_moves, test_moves = train_test_split(final_moves, test_size=0.1)
        sizes = [[5], [2, 2, 2], [5, 5], [8, 8]]
        val = SingleValueFunction(sizes[s])
    else:
        train_moves, test_moves = train_test_split(moves, test_size=0.1)
        sizes = [[5], [2, 2, 2], [5, 5, 5, 5, 5], [8, 8, 8, 8]]
        val = SelectionValueFunction(sizes[s])

    coeff = np.random.normal(size=val.coeff_len) * 0.1
    n = 0
    while 53 * n < len(coeff):
        coeff[53 * n:(53 * n + 10)] = 0
        n += 1

    val.set_coeff(coeff)

    n = 0
    train_diff = np.zeros(len(train_moves))
    train_base = copy(train_diff)
    test_diff = np.zeros(len(test_moves))
    test_base = copy(test_diff)

    timed = False
    epoch = 0
    trainerr = []
    testerr = []
    while True:
        print('entering calibration next epoch...')
        dcoeff = np.zeros(val.coeff_len)
        random.shuffle(train_moves)
        # just run one batch
        for m, move in enumerate(train_moves[:1000]):
            if train_final_scores:
                thisvalue = val(input_vec=move['game'], pos=move['pos'], mask=move['game'])
                diff = move['score'] - thisvalue
                train_base[m] = move['score']
            else:
                scores, inds = nice_allscores(move['allscores'])
                thisvalue = val(input_vec=move['game'], pos=move['pos'], indices=inds, mask=move['game'])
                diff = scores - thisvalue
                train_base[m] = np.linalg.norm(scores - scores.mean())
            train_diff[m] = np.linalg.norm(diff)

            # square_print(nn(inp))
            # print(np.linalg.norm(diff))
            gr = val.nn.grad()
            # gr = gr /(1 + np.linalg.norm(gr))
            delta = gr.dot(diff).transpose() * 0.01
            # print(diff,dcoeff,delta)
            try:
                dcoeff += delta
            except:
                dcoeff += delta[0]

            if m % 20 == 0:
                # print(np.linalg.norm(dcoeff), np.linalg.norm(val.nn.get_coeff()))
                # dnorm = np.linalg.norm(dcoeff)
                # cnorm = np.linalg.norm(val.nn.get_coeff())
                # dcoeff = math.sqrt(cnorm/dnorm)*dcoeff
                # if m%50 == 0 and m>0:
                # print(np.linalg.norm(dcoeff), np.linalg.norm(val.nn.get_coeff()))
                # print('test: ',np.linalg.norm(train_diff)/np.linalg.norm(train_base))
                coeff = val.nn.get_coeff() + dcoeff
                # print(coeff.shape)
                val.nn.set_coeff(coeff)
                dcoeff = np.zeros(val.coeff_len)
                # if m%100 == 0
        random.shuffle(test_moves)
        for m, move in enumerate(test_moves[:500]):
            if train_final_scores:
                thisvalue = val(input_vec=move['game'], pos=move['pos'], mask=move['game'])
                diff = move['score'] - thisvalue
                test_base[m] = move['score']
            else:
                scores, inds = nice_allscores(move['allscores'])
                thisvalue = val(input_vec=move['game'], pos=move['pos'], indices=inds, mask=move['game'])
                diff = scores - thisvalue
                test_base[m] = np.linalg.norm(scores - scores.mean())

            test_diff[m] = np.linalg.norm(diff)

        print(np.linalg.norm(dcoeff), np.linalg.norm(val.nn.get_coeff()))
        trainerr.append(1 - (np.linalg.norm(train_diff) / np.linalg.norm(train_base)) ** 2)
        testerr.append(1 - (np.linalg.norm(test_diff) / np.linalg.norm(test_base)) ** 2)
        print('train: ', trainerr[-1])
        print('test: ', testerr[-1])
        epoch += 1
        with open('data/calibrated_' + str(s) + '_epoch_' + str(epoch) + '.pickle', 'wb') as handle:
            pickle.dump({'function': val, 'trainerr' : trainerr, 'testerr': testerr}, handle)