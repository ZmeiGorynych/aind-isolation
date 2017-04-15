from tournament import tournament
from reporting import Reporting, get_depths
from multiprocessing import Pool
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

CUSTOM_ARGS = {"method": 'minimax', 'iterative': False, "search_depth": 6}

my_part_x2 = Agent(CustomPlayerComp(score_fn=partition_score_x2, **CUSTOM_ARGS),
                   "Partitioning with two steps")
my_x3 = Agent(CustomPlayerComp(score_fn=improved_score_fast_x3, **CUSTOM_ARGS),
      "improved, three steps exact")
my_x1 = Agent(CustomPlayerComp(score_fn=improved_score_fast, **CUSTOM_ARGS),
      "Faster improved")                        

my_x2 = Agent(CustomPlayerComp(score_fn=improved_score_fast_x2,**CUSTOM_ARGS), 
                  "improved, two steps exact, with reporting")

my_null = Agent(CustomPlayerComp(score_fn=null_score,method = 'minimax', iterative = True),
                       "Null score minimax ID")

policy_6 = SimplePolicy(6,  improved_score_fast_x2 )
policy_5 = SimplePolicy(5,  improved_score_fast_x2 )
policy_3 = SimplePolicy(3,  improved_score_fast_x2 )

my_policy_x2_5 = Agent(CustomPlayerComp(score_fn=improved_score_fast_x2,
                                        policy = policy_5, **CUSTOM_ARGS), 
                  "simple policy, max 5 moves")

my_policy_x2_3 = Agent(CustomPlayerComp(score_fn=improved_score_fast_x2,
                                        policy = policy_3, **CUSTOM_ARGS), 
                  "simple policy, max 3 moves")
my_policy_x2_6 = Agent(CustomPlayerComp(score_fn=improved_score_fast_x2,
                                        policy = policy_6, **CUSTOM_ARGS), 
                  "simple policy, max 6 moves")

#test_agents = [my_policy_x2]#my_null,my_x1, my_x2 , my_x3, my_part_x2]

#test_agents = [my_x1, my_x2 , my_x3,my_x1, my_x2 , my_x3]#,my_policy_x2_3] #]#, my_policy_x2_5, my_policy_x2_6, , my_part_x2

test_agents = [my_x3,my_x3]
def par_tournament(agent):
    result =tournament(num_matches = 10, test_agents = [agent], time_limit=float('inf'))
    depths = get_depths(result, [agent], lambda x: {'depth':x['depth'],'score': x['score'],
            'game': x['game'],'pos': x['pos'], 'simple_score':x['simple_score'],
            'move': x['move'], 'allscores': x['allscores']})
    return depths

if __name__ == '__main__':
    for i in range(1,100):
        with Pool(6) as p:
            result = p.map(par_tournament, test_agents)
            #result = []
            #for agent in test_agents:
            #    result.append(par_tournament(agent))
            with open('result_a' + str(i) + '.pickle', 'wb') as handle:
                pickle.dump(result, handle)
