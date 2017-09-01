import tensorflow as tf
import numpy as np
from neural.neural_ import get_move_mapping_tensors

conv_map, bias_map = get_move_mapping_tensors()
num_coeffs = conv_map.shape[2]
num_biases = bias_map.shape[1]
num_fields = bias_map.shape[0]

def ch_convolve_by_moves(in_fields, mask, this_conv_coeffs, biases, sess=None):
    '''
    in_fields is a batch of tensors [batch_size, num_fields, num_channels]
    mask is a tensor of 0s and 1s [num_fields]
    so conv_coeffs must have size [num_coeffs, channels_in, channels_out]
    '''
    conv_mapping = tf.constant(conv_map, dtype=tf.float32, name='conv_mapping')
    bias_mapping = tf.constant(bias_map, dtype=tf.float32, name='bias_mapping')

    input_dim = in_fields.get_shape().as_list()
    if len(input_dim) == 2:  # just batch and fields
        in_fields = tf.expand_dims(in_fields, 2)  # add the channel dimension
    # inputs conv_mapping[i,j,k], this_conv_coeffs[k,m,l], output tmp[i,j,m,l]
    tmp = tf.tensordot(conv_mapping, this_conv_coeffs, [[2], [0]])
    # print(sess.run([tf.shape(tmp), tf.shape(in_fields)]))
    # in_fields[b,j,m], output is tmp2[b,i,l]
    tmp2 = tf.tensordot(tf.cast(in_fields, tf.float32), tmp, [[1, 2], [1, 2]])

    # bias_mapping[i,k], biases[k,l], bias_term should be [b,i,l] but in this line just get [i,l]:
    tmp_bias = tf.tensordot(bias_mapping, biases, [[1], [0]])

    out = tmp2 + tf.expand_dims(tmp_bias, 0)  # use broadcasting to add biases to each batch

    if mask is not None:
        # mask is [b,i], batches x num_fields, need to apply to all channels of output - use broadcasting
        if len(mask.get_shape()) == 2:
            mask = tf.expand_dims(mask, 2)
        out = tf.multiply(out, tf.cast(mask, tf.float32))
    return out


# this wrapper just defines the Variables, to be replaced by a Keras wrapper
# approx avg number of inputs is 6 or so, so normalize init weights accordingly
def ch_convolve_by_moves_with_coeffs(in_fields, mask, out_channels, wgt_init=None, sess=None):
    in_channels = in_fields.get_shape().as_list()[-1]
    # print(sess.run(tf.shape(in_fields)[-1]))
    # print(in_fields.get_shape().as_list())
    if not wgt_init:
        wgt_init = tf.truncated_normal(shape=[num_coeffs, in_channels, out_channels], stddev=1 / 2.5,
                                       dtype=tf.float32)
    this_conv_coeffs = tf.Variable(wgt_init)

    biases = tf.Variable(np.zeros([num_biases, out_channels]), dtype=tf.float32)
    return ch_convolve_by_moves(in_fields, mask, this_conv_coeffs, biases)


def get_random_index(x, ind, batch_size, sess=None):
    '''
    x: [batch_size, num_fields]
    ind: [batch_size,2] # 2 random indexes (me and opponent) I want to grab in the corresponding row
    '''
    #
    batch_ind = tf.constant(np.array(range(batch_size))[:, None])
    batch_nums = tf.cast(batch_ind, tf.int32)
    ind = tf.cast(ind, tf.int32)
    ind1 = tf.slice(ind, [0, 0], [-1, 1])
    ind2 = tf.slice(ind, [0, 1], [-1, 1])
    ind_ext1 = tf.concat([batch_nums, ind1], 1)
    ind_ext2 = tf.concat([batch_nums, ind2], 1)
    out1 = tf.expand_dims(tf.gather_nd(x, ind_ext1), 2)
    out2 = tf.expand_dims(tf.gather_nd(x, ind_ext2), 2)
    out = tf.concat([out1, out2], 2)
    print(sess.run(tf.shape(out)))
    return out

def conv_stack(inputs, num_layers, sess=None):
    '''
    in_fields: [batch_size, num_fields]
    num_layers: int
    my_pos: [batch_size, 1]
    other_pos: [batch_size, 1]
    '''
    in_fields = tf.slice(inputs, [0, 0], [-1, num_fields])
    player_pos = tf.slice(inputs, [0, num_fields], [-1, 2])
    #     print(sess.run(in_fields))
    #     print(sess.run(tf.shape(my_pos)))
    #     print(sess.run(tf.shape(other_pos)))
    mask = in_fields
    out = tf.expand_dims(in_fields, 2)  # add the channel dimension
    for _ in range(num_layers):
        out = ch_convolve_by_moves_with_coeffs(out, in_fields, 3, sess=sess)

    player_wgt = tf.Variable(tf.truncated_normal(shape=[1, 2], dtype=tf.float32))
    # other_wgt = tf.Variable(tf.truncated_normal(shape =[1],dtype=tf.float32))

    batch_size = out.shape[0]
    return player_wgt * get_random_index(out, player_pos, batch_size, sess)
