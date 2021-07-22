import tensorflow as tf

def se_block(residual, name, ratio=8):
  """Contains the implementation of Squeeze-and-Excitation(SE) block.
  As described in https://arxiv.org/abs/1709.01507.
  """

  kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
  bias_initializer = tf.constant_initializer(value=0.0)

  with tf.variable_scope(name):
    channel = residual.get_shape()[-1]
    # Global average pooling
    squeeze = tf.reduce_mean(residual, axis=[1,2], keepdims=True)
    assert squeeze.get_shape()[1:] == (1,1,channel)
    excitation = tf.layers.dense(inputs=squeeze,
                                 units=channel//ratio,
                                 activation=tf.nn.relu,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 name='bottleneck_fc')
    assert excitation.get_shape()[1:] == (1,1,channel//ratio)
    excitation = tf.layers.dense(inputs=excitation,
                                 units=channel,
                                 activation=tf.nn.sigmoid,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 name='recover_fc')
    assert excitation.get_shape()[1:] == (1,1,channel)
    # top = tf.multiply(bottom, se, name='scale')
    scale = residual * excitation
  return scale

def COLD_fc(deep_sparse_embedding, name_pre, batch_size):
    """
    COLD model
    :param deep_sparse_embedding: [-1, 1, embedding_dim, number_of_features]
    :param name_pre: str
    :return: CTR prediction
    """
    deep_se_emb = se_block(deep_sparse_embedding, name_pre, 8)
    se_emb_reshape = tf.reshape(deep_se_emb, [batch_size, -1])
    se_emb_2 = tf.layers.dense(se_emb_reshape, 1024, name="result_projector1", activation='selu')
    se_emb_3 = tf.layers.dense(se_emb_2, 512, name="result_projector2", activation='selu')
    se_emb_4 = tf.layers.dense(se_emb_3, 256, name="result_projector3", activation='selu')
    se_emb_5 = tf.layers.dense(se_emb_4, 128, name="result_projector4", activation='selu')
    se_emb_6 = tf.layers.dense(se_emb_5, 64, name="result_projector5", activation='selu')
    logits = tf.layers.dense(se_emb_6, 1, name="result_projector6")
    return tf.nn.sigmoid(logits)