
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=False)



tf.reset_default_graph()

#rnn参数
lr = 0.01
training_iters = 100000
batch_size = 128
n_inputs = 28   # MNIST data input (img shape: 28*28)
n_steps = 28    # time steps
n_hidden_units = 128   # neurons in hidden layer
n_classes = 10      # MNIST classes (0-9 digits)

#rnn参数设置完毕

#rnn input
X = tf.placeholder("float", [None, 784])
x = tf.reshape(X,[-1, n_steps, n_inputs])
#x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
#rnn inputend
#define weight
# Define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.truncated_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.truncated_normal([n_hidden_units, 784]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[784, ]))
}
#defineend

def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    ##########################################

    # basic LSTM Cell.
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # # or
    # unpack to list [(batch, outputs)..] * steps
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)

    return results






print("helo")

decoder_op = RNN(x, weights, biases)
# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X
# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
print("ca")
# Launch the graph
# =============================================================================
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #total_batch = int(mnist.train.num_examples/batch_size)
    for epoch in range(50):
        # Loop over all batches
        for i in range(200):

            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % 1 == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))
    print("Optimization Finished!")
    # # Applying encode and decode over test set
    #test

    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:128]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(10):

        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    plt.show()

# =============================================================================

#run test imageerror
# =============================================================================
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     encode_decode = sess.run(cost, feed_dict={X: mnist.test.images[:10]})
# =============================================================================
