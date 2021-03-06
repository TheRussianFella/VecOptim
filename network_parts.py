import tensorflow as tf


def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

## Classifier

def classifier(z, num_classes):
    dense = tf.layers.dense(z, 32, activation=tf.nn.relu)
    dense = tf.layers.dense(dense, 16, activation=tf.nn.relu)
    return tf.layers.dense(dense, num_classes)


## VAE


### Encoder

def encoder(x, vec_size):
    # Convolutions 

    conv = tf.layers.conv2d(x, 16, 4, activation=tf.nn.relu)
    pool = tf.layers.max_pooling2d(conv, 6, 1)

    conv = tf.layers.conv2d(pool, 32, 4, activation=tf.nn.relu)
    pool = tf.layers.max_pooling2d(conv, 8, 2)

    # Denses

    pool_shape = pool.shape

    flat_pool = tf.reshape(pool, [-1, (pool_shape[1]*pool_shape[2]*pool_shape[3]).value])

    dense = tf.layers.dense(flat_pool, 256, activation=tf.nn.relu)

    # Output

    gausian_params = tf.layers.dense(dense, vec_size*2)

    return gausian_params[:, :vec_size], 1e-16 + tf.nn.softplus(gausian_params[:, vec_size:])


### Decoder 
def decoder(z, vec_size, shape, num_channels=1):
    
    
    decoder_weights = {
        'first_dense': tf.Variable(glorot_init(shape=[vec_size, vec_size*2]), dtype=tf.float32, name='first_dense_weights'),
        'second_dense': tf.Variable(glorot_init(shape=[vec_size*2, vec_size*3]), dtype=tf.float32, name='second_dense_weights'),
        'third_dense': tf.Variable(glorot_init(shape=[vec_size*3, vec_size*4]), dtype=tf.float32, name='third_dense_weights'),
        'output': tf.Variable(glorot_init(shape=[vec_size*4, shape[0]*shape[1]*num_channels]), dtype=tf.float32, name='output_weights'),
    }

    decoder_biases = {
        'first_dense': tf.Variable(tf.zeros(vec_size*2), dtype=tf.float32, name='first_dense_bias'),
        'second_dense': tf.Variable(tf.zeros(vec_size*3), dtype=tf.float32, name='second_dense_bias'),
        'third_dense': tf.Variable(tf.zeros(vec_size*4), dtype=tf.float32, name='third_dense_bias'),
        'output': tf.Variable(tf.zeros(shape[0]*shape[1]*num_channels), name='output_bias')
    }
    dense = tf.nn.relu(tf.matmul(z, decoder_weights['first_dense']) + decoder_biases['first_dense'])
    
    dense = tf.nn.relu(tf.matmul(dense, decoder_weights['second_dense']) + decoder_biases['second_dense'])
    
    dense = tf.nn.relu(tf.matmul(dense, decoder_weights['third_dense']) + decoder_biases['third_dense'])
    
    raw_output = tf.matmul(dense, decoder_weights['output']) + decoder_biases['output']
    
    raw_output = tf.clip_by_value(raw_output, 0, 255)
    
    return tf.reshape(raw_output, shape=[tf.shape(raw_output)[0], shape[0], shape[1], num_channels])