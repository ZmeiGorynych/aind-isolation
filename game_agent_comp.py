"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
from value_functions import partition_score_x2, partition_score, partition, \
    generate_all_moves

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

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

    return improved_score_fast_x2(game, player)



def improved_score_fast_x2(game, player):
    try:
        move_dict = player.move_dict
    except: #if we're called inside a test without a properly initialized player
        return 0.0

    own_moves = fast_legal_moves(player, game, move_dict)
    opp_moves = fast_legal_moves(game.get_opponent(player), game, move_dict)

    is_winner = (player == game.inactive_player) and len(opp_moves) == 0
    is_loser = (player == game.active_player) and len(own_moves) == 0

    if is_loser:
        return float("-inf")

    if is_winner:
        return float("inf")

    own_moves_x2 = set(own_moves)
    for move in own_moves:
        x2 = fast_legal_moves_from_move(move, game, move_dict)
        for x in x2:
            own_moves_x2.add(x)
    opp_moves_x2 = set(opp_moves)
    for move in opp_moves:
        x2 = fast_legal_moves_from_move(move, game, move_dict)
        for x in x2:
            opp_moves_x2.add(x)

    return float(len(own_moves_x2) - 2*len(opp_moves_x2))

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

def open_fast_x2_naive(game, player):

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

    return float(own_moves_x2)

def open_fast_x2(game, player):
    move_dict = player.move_dict
    own_moves = fast_legal_moves(player, game, move_dict)
    opp_moves = fast_legal_moves(game.get_opponent(player), game, move_dict)

    is_winner = (player == game.inactive_player) and len(opp_moves) == 0
    is_loser = (player == game.active_player) and len(own_moves) == 0

    if is_loser:
        return float("-inf")

    if is_winner:
        return float("inf")

    own_moves_x2 = set(own_moves)
    for move in own_moves:
        x2 = fast_legal_moves_from_move(move, game, move_dict)
        for x in x2:
            own_moves_x2.add(x)

    return float(len(own_moves_x2))

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

class CustomPlayerComp:
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

    def __init__(self, search_depth=3, score_fn=improved_score_fast_x2,
                 iterative=True, method='minimax', timeout=10., policy = None):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.move_dict = generate_all_moves()
        self.board_cache = [0]*49
        self.policy = policy

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

        moves = game.get_legal_moves(self)

        # why would I select a first move from an opening book rather than straight?
            # loc = game.get_player_location()
            # if not loc:
            # TODO: better first move?
        info = {}
        info['depth'] = 0

        if not len(moves):
            return ((-1,-1), info)


        if self.score == partition_score or self.score == partition_score_x2: # don't need this data otherwise
            partition_score_ = partition(game, self.move_dict,
                               game.get_player_location(self),
                               game.get_player_location(game.get_opponent(self)))

            if partition_score_ is not 0:
                self.game_partitioned = True
            else:
                self.game_partitioned = False

        depth = 0
        self.allscores = None
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
                while depth < game.height*game.width:
                    #print(depth, self.time_left())
                    depth += 1
                    #if self.method=='minimax':
                    score, bestmove = func(game,depth,maximizing_player=True, top_level=True)
                    if score == float('inf') or score == float('-inf'): # found a strict win
                        break
            else:
                score, bestmove = func(game, self.search_depth, maximizing_player=True, top_level=True)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            #print('depth reached: ', depth)
            pass

        if self.iterative:
            info['depth'] = depth
        else:
            info['depth'] = self.search_depth

        info['score'] = score
        info['allscores'] = self.allscores

        # Return the best move from the last completed search iteration
        return (bestmove, info)
        #raise NotImplementedError

    def minimax(self, game, depth, maximizing_player=True, top_level = False):
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
        moves = game.get_legal_moves(game.active_player)
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
        if top_level:
            self.allscores = []
        for move in moves:
            thisgame = game.forecast_move(move)
            thisscore = self.minimax(thisgame, depth-1, not maximizing_player)
            if top_level:
                self.allscores.append((thisscore[0], move))
            else:
                if (maximizing_player and thisscore[0] == float('inf')) or \
                        (not maximizing_player and thisscore[0] == float('-inf')):
                    return (thisscore[0], move) # already won/lost
            scores.append((thisscore[0], move))
            if maximizing_player:
                best_score = max(scores, key=lambda x: x[0])
            else:
                best_score = min(scores, key=lambda x: x[0])
            scores = [best_score]
        if top_level:
            pass

        return best_score



    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True, top_level =False):
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

        # if the game is over
        moves = game.get_legal_moves(game.active_player)
        if not len(moves):  # if no legal moves
            if self == game.active_player:
                return (float('-inf'), (-1, -1))
            else:
                return (float('inf'), (-1, -1))

        if depth == 0:
            return (self.score(game, self), None) # so convention is score from my perspective
        # get legal moves

        # spawn a new board for each possible legal move
        scores = []
        if self.policy is None:
            for move in moves:
                thisgame = game.forecast_move(move)
                thisscore = self.alphabeta(thisgame, depth-1, alpha, beta, not maximizing_player)
                scores.append((thisscore[0], move))
                if maximizing_player:
                    best_score = max(scores, key=lambda x: x[0])
                    if best_score[0] >= beta:
                        return best_score
                    alpha = max(alpha, best_score[0])
                else:
                    best_score = min(scores, key=lambda x: x[0])
                    if best_score[0] <= alpha:
                        return best_score
                    beta = min(beta, best_score[0])

                scores = [best_score]
        else:
            potential_moves = []
            for move in moves:
                potential_moves.append((move,game.forecast_move(move)))
            if len(potential_moves) > self.policy.max_moves:
                newmoves = self.policy(potential_moves,self,maximizing_player)
            else:
                newmoves = potential_moves

            for newmove in newmoves:
                move, thisgame = newmove
                thisscore = self.alphabeta(thisgame, depth - 1, alpha, beta, not maximizing_player)
                scores.append((thisscore[0], move))
                if maximizing_player:
                    best_score = max(scores, key=lambda x: x[0])
                    if best_score[0] >= beta:
                        return best_score
                    alpha = max(alpha, best_score[0])
                else:
                    best_score = min(scores, key=lambda x: x[0])
                    if best_score[0] <= alpha:
                        return best_score
                    beta = min(beta, best_score[0])

                scores = [best_score]

        return best_score
