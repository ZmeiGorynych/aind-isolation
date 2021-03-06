"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def generate_all_moves():
    from sample_players import RandomPlayer
    from isolation import Board

    move_dict = {}
    player1 = RandomPlayer()
    player2 = RandomPlayer()
    game = Board(player1, player2)
    all_moves = game.get_legal_moves(player1)

    for move in all_moves:
        new_game = game.forecast_move(move)
        move_dict[move] = set(new_game.get_legal_moves(player1))
        #print(len(move_dict[move]))
    return move_dict


def partition(game, move_dict, my_pos, other_pos=None):
    this_component_size = 1
    prev_nodes = {}
    latest_nodes = set([my_pos])
    while len(latest_nodes):
        next_nodes = set()
        for x in latest_nodes:
            # what are the moves we can get to from the current boundary?
            for move in move_dict[x]:
                if move == other_pos:
                    return 0  # we're in the same component of the board
                if move not in prev_nodes and game.move_is_legal(move):
                    next_nodes.add(move)
        prev_nodes = latest_nodes
        latest_nodes = next_nodes
        this_component_size += len(latest_nodes)
        #print(this_component_size)

    # if we got this far, the two players are in unconnected components of the board
    if other_pos is None:
        return this_component_size
    else:
        return this_component_size - partition(game, move_dict, other_pos)


def partition2(game, move_dict, my_pos, other_pos):
    this_component_size = 1
    prev_nodes = {}
    latest_nodes = set([my_pos])
    while len(latest_nodes):
        next_nodes = set()
        for x in latest_nodes:
            # what are the moves we can get to from the current boundary?
            for move in move_dict[x]:
                if move == other_pos:
                    return 0  # we're in the same component of the board
                if move not in prev_nodes and game.move_is_legal(move):
                    next_nodes.add(move)
        prev_nodes = latest_nodes
        latest_nodes = next_nodes
        this_component_size += len(latest_nodes)
        #print(this_component_size)

    # if we got this far, the two players are in unconnected components of the board
    other_component_size = 1
    prev_nodes = {}
    latest_nodes = set([other_pos])
    while len(latest_nodes):
        next_nodes = set()
        for x in latest_nodes:
            # what are the moves we can get to from the current boundary?
            for move in move_dict[x]:
                if move == other_pos:
                    return 0  # we're in the same component of the board
                if move not in prev_nodes and game.move_is_legal(move):
                    next_nodes.add(move)
        prev_nodes = latest_nodes
        latest_nodes = next_nodes
        other_component_size += len(latest_nodes)

    return this_component_size - other_component_size

def improved_score_fast_x2_naive(game, player):

    move_dict = player.move_dict
    own_moves = fast_legal_moves(player, game, move_dict)
    opp_moves = fast_legal_moves(game.get_opponent(player), game, move_dict)

    is_winner = (player == game.inactive_player) and len(opp_moves) == 0
    is_loser = (player == game.active_player) and len(own_moves) == 0

    if is_loser:
        return float("-inf")

    if is_winner:
        return float("inf")

    own_moves_x2 = 0
    for move in own_moves:
        own_moves_x2 += num_legal_moves_from_move(move, game, move_dict)
    opp_moves_x2 = 0
    for move in opp_moves:
        opp_moves_x2 += num_legal_moves_from_move(move, game, move_dict)

    return float(own_moves_x2 - 2*opp_moves_x2)

def partition_score_2(game, player):
    """The "Improved" evaluation function discussed in lecture that outputs a
    score equal to the difference in the number of moves available to the
    two players.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """

    if player.game_partitioned:
        return improved_score_fast_x2_naive(game,player)

    move_dict = player.move_dict
    own_moves = num_legal_moves(player, game, move_dict)
    opp_moves = num_legal_moves(game.get_opponent(player), game, move_dict)

    is_winner = (player == game.inactive_player) and opp_moves == 0
    is_loser = (player == game.active_player) and own_moves == 0

    if is_loser:
        return float("-inf")

    if is_winner:
        return float("inf")

    if game.move_count >=8 : # arbitrary attempt at optimization
        score1 = partition(game, player.move_dict,
                           game.get_player_location(player),
                           game.get_player_location(game.get_opponent(player)))

        if score1 != 0:
            #print('partition detected!', game.move_count, 100* score1 )
            return 100 * score1


    return float(own_moves - 2*opp_moves)

def custom_score_1(game, player):
    """The "Improved" evaluation function discussed in lecture that outputs a
    score equal to the difference in the number of moves available to the
    two players.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - 2*opp_moves)

def improved_score_fast(game, player):

    move_dict = player.move_dict
    own_moves = num_legal_moves(player, game, move_dict)
    opp_moves = num_legal_moves(game.get_opponent(player), game, move_dict)

    is_winner = (player == game.inactive_player) and opp_moves == 0
    is_loser = (player == game.active_player) and own_moves == 0

    if is_loser:
        return float("-inf")

    if is_winner:
        return float("inf")

    return float(own_moves - 2*opp_moves)

def improved_score_fast_x2(game, player):

    move_dict = player.move_dict
    own_moves = fast_legal_moves(player, game, move_dict)
    opp_moves = fast_legal_moves(game.get_opponent(player), game, move_dict)

    is_winner = (player == game.inactive_player) and len(opp_moves) == 0
    is_loser = (player == game.active_player) and len(own_moves) == 0

    if is_loser:
        return float("-inf")

    if is_winner:
        return float("inf")

    own_moves_x2 = 0
    for move in own_moves:
        own_moves_x2 += num_legal_moves_from_move(move, game, move_dict)
    opp_moves_x2 = 0
    for move in opp_moves:
        opp_moves_x2 += num_legal_moves_from_move(move, game, move_dict)

    return float(own_moves_x2 - opp_moves_x2)

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    move_dict = player.move_dict
    own_moves = num_legal_moves(player, game, move_dict)

    return own_moves
    #raise NotImplementedError


def num_legal_moves(player, game, move_dict):
    move = game.get_player_location(player)
    return num_legal_moves_from_move(move, game, move_dict)

def num_legal_moves_from_move(move, game, move_dict):
    possibles = move_dict[move]
    n=0
    for x in possibles:
        if game.move_is_legal(x):
            n+=1
    return n

def fast_legal_moves(player, game, move_dict):
    move=game.get_player_location(player)
    return fast_legal_moves_from_move(move, game, move_dict)

def fast_legal_moves_from_move(move, game, move_dict):
    possibles = move_dict[move]
    return [x for x in possibles if game.move_is_legal(x)]

class CustomPlayer3:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)  This parameter should be ignored when iterative = True.

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).  When True, search_depth should
        be ignored and no limit to search depth.

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score_1,
                 iterative=True, method='minimax', timeout=20.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.move_dict = generate_all_moves()
        self.cached_tree = {}





    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            DEPRECATED -- This argument will be removed in the next release

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left
        #self.moves += 1

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        # check if I have an initial position?

        moves = fast_legal_moves(self, game, self.move_dict)

        # why would I select a first move from an opening book rather than straight?
            # loc = game.get_player_location()
            # if not loc:
            # TODO: better first move?
        if not len(moves):
            return (-1,-1)

        if self.score == partition_score_2: # don't need this data otherwise
            partition_score_ = partition(game, self.move_dict,
                               game.get_player_location(self),
                               game.get_player_location(game.get_opponent(self)))

            if partition_score_ is not 0:
                self.game_partitioned = True
                self.cached_tree = {} # will move to a different heuristic now so need to reset cache
            else:
                self.game_partitioned = False


        depth = 0
        try:
            if self.method == 'minimax':
                func = self.minimax
            elif self.method == 'alphabeta':
                func = self.alphabeta
            else:
                raise ValueError('The method is not minimax or alphabeta, don\'t understand ' + self.method)
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                if game.move_count <= 3:
                    self.cached_tree = {}
                if len(self.cached_tree):
                    # look up the last two moves and get that cached branch:
                    my_prev_move = game.get_player_location(self)
                    opp_prev_move = game.get_player_location(game.get_opponent(self))
                    if 'my_prev_move' in self.cached_tree and 'opp_prev_move' in self.cached_tree['my_prev_move']:
                        self.cached_tree = self.cached_tree[my_prev_move][opp_prev_move]
                    #except:
                        #raise ValueError('something went wrong with the cache?')
                        #self.cached_tree = {}
                while True:
                    #print(depth, self.time_left())
                    depth += 1
                    #if self.method=='minimax':
                    score, bestmove = func(game,depth,maximizing_player=True)
            else:
                score, bestmove = func(game, self.search_depth, maximizing_player=True)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            #print('depth reached: ', depth)
            pass

        # Return the best move from the last completed search iteration
        #print(bestmove)
        return bestmove
        #raise NotImplementedError

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        # if the game is over
        moves = fast_legal_moves(game.active_player, game, self.move_dict)
        if not len(moves):  # if no legal moves
            if self == game.active_player:
                return (float('-inf'), (-1, -1))
            else:
                return (float('inf'), (-1, -1))

        # so convention is score from my perspective
        # the move would be the active player's position, but why bother?

        if depth == 0:
            return (self.score(game, self),None)

        # TODO: finish this function!
        # spawn a new board for each possible legal move
        scores = []
        for move in moves:
            thisgame = game.forecast_move(move)
            thisscore = self.minimax(thisgame, depth-1, not maximizing_player)
            scores.append((thisscore[0], move))
            if maximizing_player:
                best_score = max(scores, key=lambda x: x[0])
            else:
                best_score = min(scores, key=lambda x: x[0])
            scores = [best_score]

        return best_score



    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True, cached_tree = None):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        #print('entering alphabeta')
        if alpha > beta:
            print('alpha > beta!')
            #pass #raise ValueError('alpha bigger than beta!')

        if self.time_left() < self.TIMER_THRESHOLD:
            #print('timeout!')
            raise Timeout()

        if cached_tree is None:
            cached_tree = self.cached_tree

        # if the game is over
        moves = fast_legal_moves(game.active_player, game, self.move_dict)
        if not len(moves):  # if no legal moves
            if self == game.active_player:
                score = (float('-inf'), (-1, -1))
            else:
                score = (float('inf'), (-1, -1))

            #cached_tree['score'] = score
            return score


        # if we didn't win or lose:
        if depth == 0:
            score = (self.score(game, self), None)
            #cached_tree['score'] = score
            return score

        # TODO: finish this function!
        # get legal moves

        # spawn a new board for each possible legal move
        scores = []
        for move in moves:
            thisscore = None
            if move in cached_tree and 'score' in cached_tree[move]:
                cached_score = cached_tree[move]['score']
                if (cached_score[0] == float('inf') and maximizing_player) \
                        or (cached_score[0] == float('-inf') and not maximizing_player):
                    thisscore = cached_score # no need to re-run known winning or losing branches
                    return thisscore
            else:
                cached_tree[move] = {}

            if thisscore is None:
                thisgame = game.forecast_move(move)
                thisscore = self.alphabeta(thisgame, depth-1, alpha, beta, not maximizing_player, cached_tree[move])

            if thisscore[0] == (float('-inf'), (-1, -1)) or thisscore[0] == (float('inf'), (-1, -1)):
                cached_tree[move]['score'] = (thisscore[0], move) # cache actual wins or losses

            scores.append((thisscore[0], move))

            if maximizing_player:
                best_score = max(scores, key=lambda x: x[0])
                if best_score[0] >= beta:
                    #cached_tree['score'] = best_score
                    return best_score
                alpha = max(alpha, best_score[0])
            else:
                best_score = min(scores, key=lambda x: x[0])
                if best_score[0] <= alpha:
                    #cached_tree['score'] = best_score
                    return best_score
                beta = min(beta, best_score[0])

            scores = [best_score]

        #cached_tree['score'] = best_score

        return best_score
