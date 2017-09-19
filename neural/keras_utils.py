from math import exp
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import TruncatedNormal
from keras.regularizers import l2
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Flatten, Dense, Activation, Dropout, Conv1D, Multiply,Reshape
from keras.layers.merge import Concatenate, Add
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from constants import BOARD_SIZE
from neural.tensorflow_utils import ch_convolve_by_moves, num_biases, num_coeffs, num_fields

class ConvByMoveLayer(Layer):
    def __init__(self, out_channels=None, mask=None, l2_reg = 0.01, **kwargs):
        self.out_channels = out_channels
        self.mask = mask
        self.l2_reg = l2_reg
        super(ConvByMoveLayer, self).__init__(**kwargs)

    # The below was intended to make this layer work with model.save(), but throws a strange error
    # def get_config(self):
    #     config = {
    #         'out_channels': self.out_channels,
    #         'mask':self.mask,
    #         'l2_reg':self.l2_reg
    #     }
    #     base_config = super(ConvByMoveLayer, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))

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
    legal_moves = Input(shape=[BOARD_SIZE, 1])
    next_move = Input(shape=[BOARD_SIZE, 1])
    mask = board_state
    out = Concatenate()([board_state, player_pos_one_hot])
    # the below is to match the number of channels to what the resnet layers expect
    out = Conv1D(filters=num_features, kernel_size=1)(out)
    for _ in range(num_res_modules):
        out = ResNetLayerFun(out, num_features, mask, drop_rate)
    out = Activation('relu')(out)
    # add player positions again, just in case #TODO is that helpful?
    out = Concatenate()([out, player_pos_one_hot, legal_moves])
    # out = ConvByMoveLayer(num_features, mask)(out)

    # share information across fields
    dense_conv = Flatten()(out)
    dense_conv = Dense(3 * BOARD_SIZE, activation='relu')(dense_conv)
    dense_conv = Dense(BOARD_SIZE)(dense_conv)
    dense_conv = Reshape([BOARD_SIZE,1])(dense_conv)
    out = dense_conv

    # # add that to the original and collapse to one channel : fits very badly
    # out = Concatenate()([out,dense_conv])
    # out = Conv1D(filters = 1, kernel_size=1, activation='relu')(out)

    # helper function, Keras doesn't seem to have one, strangely
    sum_dim1 = Lambda(lambda x: K.sum(x, axis=1), output_shape=[1])

    out_all_moves = Activation('sigmoid')(out)
    out_all_moves = Multiply()([out_all_moves, legal_moves])

    # sum_moves = Reshape([1,1])(sum_dim1(out_all_moves))
    # inv_sum_moves = Activation(lambda x: K.pow(x,-1))(sum_moves)
    # out_all_moves = Multiply()([out_all_moves, inv_sum_moves])

    # output the value corresponding to the next move
    out_next = Multiply()([out, next_move])
    out_next = sum_dim1(out_next)
    # out = Flatten()(out)
    # out = Dense(10, activation = 'relu')(out)
    # out = Dense(1, activation = activation)(out)
    out_next = Activation(activation)(out_next)
    deep_model = Model(inputs = [board_state, player_pos_one_hot, legal_moves, next_move], outputs = out_next)
    deep_Q = Model(inputs = [board_state, player_pos_one_hot, legal_moves], outputs = out_all_moves)
    return deep_model, deep_Q
