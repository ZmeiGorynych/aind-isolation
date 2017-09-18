from math import floor
BOARD_WIDTH = 4
BOARD_SIZE = BOARD_WIDTH*BOARD_WIDTH
BOARD_MID = (BOARD_WIDTH-1)/2
my_half = floor((BOARD_WIDTH + 1)/2)
# number of cells modulo the 8 board symmetries
NUM_BIASES = int(my_half*(my_half+1)/2)
