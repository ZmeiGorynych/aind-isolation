"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import copy
from value_functions import *

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass




class CustomPlayer2:
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
        self.board_cache = [0]*BOARD_SIZE





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
        else:
            bestmove = moves[0]
        a = print

        if self.score == partition_score or self.score == partition_score_x2: # don't need this data otherwise
            partition_score_ = partition(game, self.move_dict,
                               game.get_player_location(self),
                               game.get_player_location(game.get_opponent(self)))

            if partition_score_ is not 0:
                self.game_partitioned = True
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



    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
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
        moves = fast_legal_moves(game.active_player, game, self.move_dict)
        if not len(moves):  # if no legal moves
            if self == game.active_player:
                return (float('-inf'), (-1, -1))
            else:
                return (float('inf'), (-1, -1))

        if depth == 0:
            return (self.score(game, self), None) # so convention is score from my perspective

        # TODO: finish this function!
        # get legal moves

        # spawn a new board for each possible legal move
        scores = []
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

        return best_score
