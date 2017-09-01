from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import TruncatedNormal
import numpy as np
from neural.tensorflow_utils import ch_convolve_by_moves, num_biases, num_coeffs, num_fields

class ConvByMoveLayer(Layer):
    def __init__(self, out_channels, mask=None, **kwargs):
        self.out_channels = out_channels
        self.mask = mask
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
                                           trainable=True)
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