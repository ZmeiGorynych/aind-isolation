import pickle
def load_simulation_data(files):
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
    return depths

def transpose_list_of_lists(lol):
    
    transp = []

    for game in lol:
        for m, move_depth in enumerate(game):
            if len(transp) <= m:
                transp.append([])
            transp[m].append(move_depth)
    return transp

