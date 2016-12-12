import tensorflow as tf

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64
num_hidden2 = 32
image_size = 28
num_labels = 10
num_channels = 1

graph = tf.Graph()

with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    tf_test_dataset = tf.constant(testData)

    # First CONV layer variables, in truncated normal distribution.
    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))

    # dropout parameter
    keep_prob = tf.placeholder("float")

    # Second CONV layer variables
    layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

    # Three FC layer variables
    layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

    layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_hidden2], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden2]))

    layer5_weights = tf.Variable(tf.truncated_normal(
      [num_hidden2, num_labels], stddev=0.1))
    layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
    # Model.
    def model(data):
        # The convolutional model above uses convolutions with stride 2 to reduce
        # the dimensionality. Replace the strides by a max pooling operation (nn.max_pool())
        # of stride 2 and kernel size 2.
        # I will use 1 x 1 convention then max pooling.

        # layer1
        # layer1: Conv1 layer
        # patch 5 * 5 , input_channel 1, depth 16 
        # A cube of [28,28,16]
        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)

        # pooling1
        pool1 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool1')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0,   alpha=0.001 / 9.0, beta=0.75,
                        name='norm1')

        # layer2
        conv = tf.nn.conv2d(norm1, layer2_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        # pooling2
        pool2 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool1')
        norm2 = tf.nn.lrn(pool2, 4, bias=1.0,   alpha=0.001 / 9.0, beta=0.75,
                        name='norm1')

        # norm2, alpha=alpha=0.001 / 100.0;  Test accuracy: 90.3% 
        # norm2, alpha=alpha=0.1;  Test accuracy: 90.4% 

        # layer3
        conv = tf.nn.conv2d(norm2, layer2_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)



        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

        # RELU
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)

        hidden = tf.matmul(hidden, layer4_weights) + layer4_biases

        # add a dropout
        # hidden = tf.nn.dropout(hidden, keep_prob)

        result = tf.matmul(hidden, layer5_weights) + layer5_biases

        return result
  
    
    
    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Learning rate decay
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.96, staircase=True)


    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_dataset))