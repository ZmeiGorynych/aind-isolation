import pickle, glob

files = glob.glob('result*.pickle')
print(files)

depths = {}
for file in files[:4]:
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
#print(depths)

# collate all moves in one big list
moves = []
for player, games in depths.items():
    for game in games:
        for move in game:
            if move['score'] != float('inf') and move['score'] != float('-inf'):
                moves.append(move)

from neural import NNValueFunction
import numpy as np

#example = depths['simple policy, max 5 moves'][0]
#move = example[0]

val = NNValueFunction([1, 3, 3])
val.set_coeff(np.random.normal(size =val.coeff_len))
n = 0
diff = np.zeros(len(moves))
while True:
    for m, move in  enumerate(moves):
        diff[m] = move['score'] - val(input_vec = move['game'], pos = move['pos'])
        #square_print(nn(inp))
        #print(np.linalg.norm(diff))
        gr = val.nn.grad()
        gr = gr /(1 + np.linalg.norm(gr))
        coeff = val.nn.get_coeff() + gr.dot(diff[m])*0.01
        val.nn.set_coeff(coeff)

        print(np.linalg.norm(diff))
