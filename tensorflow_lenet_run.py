import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split

# Preprocessing
def reformat(dataset, labels):
    dataset = dataset.reshape(
        (-1, 28, 28, 1)).astype(np.float32)
    labels = (np.arange(10) == labels[:,None]).astype(np.float32)
    return dataset, labels

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
        / predictions.shape[0])

print "[INFO] loading MNIST..."
dataset = datasets.fetch_mldata("MNIST Original")
print "[INFO] finished loading MNIST from ~/scikit_learn_data/mldata/."
data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
data = data[:, np.newaxis, :, :]
(trainData, testData, trainLabels, testLabels) = train_test_split(
    data / 255.0, dataset.target.astype("int"), test_size=0.3)
trainData, trainLabels = reformat(trainData, trainLabels)
testData, testLabels = reformat(testData, testLabels)
print('Training set', trainData.shape, trainLabels.shape)
print('Test set', testData.shape, testLabels.shape)

# constructing the graph
batch_size = 128
patch_size = 5
depth = (20, 50)
num_hidden = (64, 32)
image_height = 28
image_width = 28
num_labels = 10
num_channels = 1

graph = tf.Graph()

with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_height, image_width, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    tf_test_dataset = tf.constant(testData)

    # First CONV layer variables, in truncated normal distribution.
    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth[0]], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros(depth[0]))

    # dropout parameter
    keep_prob = tf.placeholder("float")

    # Second CONV layer variables
    layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth[0], depth[1]], stddev=0.1))
    layer2_biases = tf.Variable(tf.zeros(depth[1]))

    # Three FC layer variables
    layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth[1], num_hidden[0]], stddev=0.1))
    layer3_biases = tf.Variable(tf.zeros(num_hidden[0]))

    layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden[0], num_hidden[1]], stddev=0.1))
    layer4_biases = tf.Variable(tf.zeors(num_hidden[1]))

    layer5_weights = tf.Variable(tf.truncated_normal(
      [num_hidden[1], num_labels], stddev=0.1))
    layer5_biases = tf.Variable(tf.zeors(num_labels))    

    # Model.
    def model(data):
        # conv layer1
        conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        hidden1 = tf.nn.relu(conv1 + layer1_biases)
        pool1 = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool1')
        # Optional: normalization
        # norm1 = tf.nn.lrn(pool1, 4, bias=1.0,   alpha=0.001 / 9.0, beta=0.75,
        #                 name='norm1')   

        # conv layer2
        conv2 = tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='SAME')
        hidden2 = tf.nn.relu(conv2 + layer2_biases)
        pool2 = tf.nn.max_pool(hidden2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool1')
        # Optional: normalization
        # norm2 = tf.nn.lrn(pool2, 4, bias=1.0,   alpha=0.001 / 9.0, beta=0.75,
        #                 name='norm1')

        # FC layer1
        shape = pool2.get_shape().as_list()
        reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden3 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)

        # FC layer2
        hidden4 = tf.matmul(hidden3, layer4_weights) + layer4_biases

        # Optional: dropout
        # hidden = tf.nn.dropout(hidden, keep_prob)

        # FC layer3
        result = tf.matmul(hidden4, layer5_weights) + layer5_biases

        return result

    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

    # Learning rate decay
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.96, staircase=True)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

num_steps = 5000

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        # stochastic gradient descent
        batch_index = np.random.choice(trainLabels.shape[0], batch_size)
        batch_data = trainData[batch_index]
        batch_labels = trainLabels[batch_index]
        # batch gradient descent
        #offset = (step * batch_size) % (trainLabels.shape[0] - batch_size)
        #batch_data = trainData[offset:(offset + batch_size), :, :, :]
        #batch_labels = trainLabels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:1.0}
        
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(session=session,feed_dict={keep_prob:1.0}), testLabels))
            
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(session=session,feed_dict={keep_prob:1.0}), testLabels))   
