%tensorflow_version 1.x
!mkdir MNIST_data

#----------------------Import required libraries-------------#
import os, time, itertools, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#-------------------Load the Data(MNIST)----------------------#
mnist = input_data.read_data_sets("MNIST_data/")
X_test = mnist.test.images # X_test shape: [num_test, 28*28]
X_test = X_test.reshape([-1, n_steps, n_inputs])
y_test = mnist.test.labels

#--------------------Initialize the hyperparameters and parameters-----------#
# hyperparameters
n_neurons = 128
learning_rate = 0.001
batch_size = 128
n_epochs = 10
# parameters
n_steps = 28 # 28 rows
n_inputs = 28 # 28 cols
n_outputs = 10 # 10 classes
# build a rnn model
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
output, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
logits = tf.layers.dense(state, n_outputs)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
prediction = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
predictions = tf.argmax(logits,1)

#-----------------------Train the model------------------------#
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    n_batches = mnist.train.num_examples // batch_size
    for epoch in range(n_epochs):
        for batch in range(n_batches):
            X_train, y_train = mnist.train.next_batch(batch_size)
            X_train = X_train.reshape([-1, n_steps, n_inputs])
            sess.run(optimizer, feed_dict={X: X_train, y: y_train})
        loss_train, acc_train = sess.run(
            [loss, accuracy], feed_dict={X: X_train, y: y_train})
        print('Epoch: {}, Train Loss: {:.3f}, Train Acc: {:.3f}'.format(
            epoch + 1, loss_train, acc_train))
    predict = sess.run(
        predictions, feed_dict={X: X_test, y: y_test})
    #print('Test Loss: {:.3f}, Test Acc: {:.3f}'.format(loss_test, acc_test))
    
#-------------------Output the predicted and actual labels of one test sample to test the authenticity-----#
print("The predicted label of index 0 is:")
print(predict[0])
print("The actual label of index 0 is: ")
print(mnist.test.labels[0])
first_image = mnist.test.images[0]
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()
print(pixels.shape)