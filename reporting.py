class Borg:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state

class Reporting(Borg):
    def __init__(self):
        Borg.__init__(self)
        if 'report' not in self.__dict__:
            self.report = []


def get_depths(report, test_agents, func=lambda x: x['depth']):

    # get player names:
    players = []
    for game in report:
        for move in game['moves']:
            try:
                full_player = [p for p in test_agents if p.player == move['active_player']][0]
                if full_player not in players:
                    players.append(full_player)
            except:
                pass

    print(players)
    depths = {}
    out_depths = {}

    for p in players:
        depths[p] = []
        for game in report:
            winner = game['winner'] == p.player
            depths[p].append([])
            for move in game['moves']:
                try:
                    move['winner'] = winner
                except:
                    pass
                try:
                    if move['active_player'] == p.player:
                        depths[p][-1].append(func(move))
                except:
                    pass
        clean = []
        for game in depths[p]:
            if game:
                clean.append(game)

        out_depths[p.name] = clean

    return out_depths
