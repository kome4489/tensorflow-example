# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import random
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters
learning_rate = 0.1
num_steps = 9

# Network Parameters
n_hidden_1 = 10 # 1st layer number of neurons
n_hidden_2 = 10 # 2nd layer number of neurons
num_input = 7 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
#X = tf.placeholder("float32", [None,num_input])
#Y = tf.placeholder("float32", [None,num_classes])
X = tf.placeholder("float32", [None,num_input,])
Y = tf.placeholder("float32", [None,num_classes,])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    #out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    return out_layer

# Construct model
logits = neural_net(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    batch_x = [[0,0,1,0,0,1,0],[1,0,1,1,1,0,1],[1,0,1,1,0,1,1],[0,1,1,1,0,1,0],[1,1,0,1,0,1,1],[1,1,0,1,1,1,1],[1,0,1,0,0,1,0],[1,1,1,1,1,1,1],[1,1,1,1,0,1,1],[1,1,1,0,1,1,1],]
    batch_y = [[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1],]
    
    #for step in range(100):
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        #sess.run(train_op, feed_dict={X: np.array(batch_x[step%10]).reshape(1,7),Y: np.array(batch_y[step%10]).reshape(1,10)})

    while True:
        print("input 0~127 for number prediction, bigger than 127 for input times.")
        print("Waiting for input...")
        name = input()
        try:
            if(int(name) > 127):
                for step in range(int(name)):
                    # batch_x, batch_y = mnist.train.next_batch(batch_size)
                    # Run optimization op (backprop)
                    #sess.run(train_op, feed_dict={X: np.array(batch_x[step%10]).reshape(1,7),Y: np.array(batch_y[step%10]).reshape(1,10)})
                    sess.run(train_op, feed_dict={X: np.array(batch_x[step%10]).reshape(-1,7),Y: np.array(batch_y[step%10]).reshape(-1,10)})

                    print(str(step) + " training for number " + str(step%10) +" completed.")
            else:
                bin_input =[int(s) for s in list(format(int(name),'07b'))]
                print("Input value :" + str(bin_input))
                print(sess.run(logits,feed_dict={X: np.array(bin_input).reshape(-1,7)}))
        except Exception as e:
            print(str(e))
            continue

