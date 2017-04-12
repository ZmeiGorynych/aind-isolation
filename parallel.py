from multiprocessing import Pool

from tournament import tournament
from reporting import Reporting, get_depths
import importlib
from collections import namedtuple
from game_agent_comp import CustomPlayerComp
from value_functions import improved_score_fast_x2,improved_score_fast,\
    improved_score_fast_x3, partition_score_x2
from sample_players import null_score
from policy import SimplePolicy
import pickle

#importlib.reload(reporting)
Agent = namedtuple("Agent", ["player", "name"])

#r = Reporting()
#r.report = []

CUSTOM_ARGS = {"method": 'alphabeta', 'iterative': True}

my_part_x2 = Agent(CustomPlayerComp(score_fn=partition_score_x2, **CUSTOM_ARGS),
                   "Partitioning with two steps")
my_x3 = Agent(CustomPlayerComp(score_fn=improved_score_fast_x3, **CUSTOM_ARGS),
      "improved, three steps exact")

test_agents = [my_part_x2, my_x3]

def par_tournament(agent):
    result =tournament(num_matches = 1, test_agents = [agent])
    depths = get_depths(result, [agent], lambda x: (x['depth'], x['score']))
    return depths

if __name__ == '__main__':
    with Pool(2) as p:
        result = p.map(par_tournament, test_agents)

        with open('result.pickle', 'wb') as handle:
            pickle.dump(result, handle)