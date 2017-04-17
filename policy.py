from value_functions import to_pair
class SimplePolicy:
    def __init__(self, max_moves, valuation_func):
        self.valuation_func = valuation_func
        self.max_moves = max_moves

    def __call__(self, potential_moves, player, maximizing_player):
        self.moves = [None] * self.max_moves#max(2, min(self.max_moves, int(0.66 * len(potential_moves))))

        for m in potential_moves:
            move, game = m
            score =self.valuation_func(game,player)
            for i,move in enumerate(self.moves):
                if maximizing_player:
                    if move is None or move[0] < score:
                        self.moves[i] = (score, m)
                        break
                else:
                    if move is None or move[0] > score:
                        self.moves[i] = (score, m)
                        break

        return [m[1] for m in self.moves if m is not None]

class SimplePolicy2:
    def __init__(self, max_moves, valuation_func):
        self.valuation_func = valuation_func
        self.max_moves = max_moves

    def __call__(self, game, player, maximizing_player, moves):
        self.moves = [None] * self.max_moves#max(2, min(self.max_moves, int(0.66 * len(potential_moves))))
        scores = self.valuation_func(game,player, moves)
        for score in scores:
            for i,move in enumerate(self.moves):
                if maximizing_player:
                    if move is None or move[0] < score[0]:
                        self.moves[i] = score
                        break
                else:
                    if move is None or move[0] > score[0]:
                        self.moves[i] = score
                        break

        return [to_pair(m[1]) for m in self.moves if m is not None]
