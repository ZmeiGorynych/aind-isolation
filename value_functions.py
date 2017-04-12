def generate_all_moves():
    from sample_players import RandomPlayer
    from isolation import Board

    move_dict = {}
    player1 = RandomPlayer()
    player2 = RandomPlayer()
    game = Board(player1, player2)
    all_moves = game.get_legal_moves(player1)
    print(len(all_moves))
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

def partition_score(game, player):
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
        return open_fast_x2_naive(game,player)

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

def partition_score_x2(game, player):
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
        return open_fast_x2_naive(game,player)

    move_dict = player.move_dict
    own_moves = fast_legal_moves(player, game, move_dict)
    opp_moves = fast_legal_moves(game.get_opponent(player), game, move_dict)

    is_winner = (player == game.inactive_player) and len(opp_moves) == 0
    is_loser = (player == game.active_player) and len(own_moves) == 0

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

    return float(len(own_moves_x2) - 2 * len(opp_moves_x2))


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

def improved_original_score_fast(game, player):

    move_dict = player.move_dict
    own_moves = num_legal_moves(player, game, move_dict)
    opp_moves = num_legal_moves(game.get_opponent(player), game, move_dict)

    is_winner = (player == game.inactive_player) and opp_moves == 0
    is_loser = (player == game.active_player) and own_moves == 0

    if is_loser:
        return float("-inf")

    if is_winner:
        return float("inf")

    return float(own_moves - opp_moves)

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

def improved_score_fast_x3(game, player, mult=None):

    move_dict = player.move_dict
    own_moves = fast_legal_moves(player, game, move_dict)
    opp_moves = fast_legal_moves(game.get_opponent(player), game, move_dict)

    is_winner = (player == game.inactive_player) and len(opp_moves) == 0
    is_loser = (player == game.active_player) and len(own_moves) == 0

    if is_loser:
        return float("-inf")

    if is_winner:
        return float("inf")

    own_moves_x3 = set(own_moves)
    for move in own_moves:
        x2 = fast_legal_moves_from_move(move, game, move_dict)
        for x in x2:
            own_moves_x3.add(x)
            x3 = fast_legal_moves_from_move(x, game, move_dict)
            for y in x3:
                own_moves_x3.add(y)

    opp_moves_x3 = set(opp_moves)
    for move in opp_moves:
        x2 = fast_legal_moves_from_move(move, game, move_dict)
        for x in x2:
            opp_moves_x3.add(x)
            x3 = fast_legal_moves_from_move(x, game, move_dict)
            for y in x3:
                opp_moves_x3.add(y)
    # if the pools 3 moves ahead don't overlap, guess that we have a partittion, give higher weight
    if mult and not len(own_moves_x3.intersection(opp_moves_x3)):
        return mult*float(len(own_moves_x3) - len(opp_moves_x3))
    else:
        return float(len(own_moves_x3) - len(opp_moves_x3))

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
