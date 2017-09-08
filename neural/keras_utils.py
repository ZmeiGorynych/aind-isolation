from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import TruncatedNormal
from keras.regularizers import l2
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Flatten, Dense, Activation, Dropout
from keras.layers.merge import Concatenate, Add
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from constants import BOARD_SIZE
from neural.tensorflow_utils import ch_convolve_by_moves, num_biases, num_coeffs, num_fields

class ConvByMoveLayer(Layer):
    def __init__(self, out_channels, mask=None, l2_reg = 0.001, **kwargs):
        self.out_channels = out_channels
        self.mask = mask
        self.l2_reg = l2_reg
        super(ConvByMoveLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        if len(input_shape) == 2:
            in_channels = 1
        else:
            in_channels = input_shape[2]

        init_std = 1 / (3 * np.sqrt(in_channels * self.out_channels))

        self.conv_coeffs = self.add_weight(name='conv_coeffs',
                                           shape=(num_coeffs, in_channels, self.out_channels),
                                           initializer=TruncatedNormal(stddev=init_std),
                                           trainable=True,
                                           regularizer = l2(self.l2_reg))
        # print((num_coeffs, in_channels, self.out_channels))
        self.biases = self.add_weight(name='biases',
                                      shape=(num_biases, self.out_channels),
                                      initializer='zeros',
                                      trainable=True)
        super(ConvByMoveLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return ch_convolve_by_moves(x, self.mask, self.conv_coeffs, self.biases)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], num_fields, self.out_channels)
    
    
def ResNetLayerFun(x, num_features = 3, mask = None, drop_rate = 0.1):
    tmp = BatchNormalization()(x)
    tmp = Activation('relu')(tmp)
    tmp = ConvByMoveLayer(num_features, mask)(tmp)
    tmp = BatchNormalization()(tmp)
    tmp = Activation('relu')(tmp)
    tmp = ConvByMoveLayer(num_features, mask)(tmp)
    tmp = Dropout(drop_rate)(tmp)
    return  Add()([x,tmp])

   
def deep_model_fun(num_features = 16, num_res_modules = 16, drop_rate = 0.1, activation = 'sigmoid'):
    player_pos_one_hot = Input(shape = [BOARD_SIZE, 2])
    board_state = Input(shape=[BOARD_SIZE,1])
    mask = board_state
    #tmp1 = K.expand_dims(board_state, 2)# TODO: do this in Keras code
    out = Concatenate()([board_state, player_pos_one_hot])
    out = ConvByMoveLayer(num_features, mask)(out)
    for _ in range(num_res_modules):
        out = ResNetLayerFun(out, num_features, mask, drop_rate)
    out = Activation('relu')(out)
    out = Concatenate()([out, player_pos_one_hot])
    out = Flatten()(out)
    out = Dense(10, activation = 'relu')(out)
    out = Dense(1, activation = activation)(out)

    deep_model = Model(inputs = [player_pos_one_hot, board_state], outputs = out)
    return deep_model
