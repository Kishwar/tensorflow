# Contact info: Kishwar Kumar [kumar.kishwar@gmail.com]
# Country: Germany
#

__author__ = 'kishwarkumar'
__date__ = '18.04.18' '21:39'

# imports
from tensorflow.examples.tutorials.mnist import input_data
from model import *

def run():

    # lets get mnist data (It may take some time, depending on internet speed)
    mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)

    # lets define the place holder for input data
    x = tf.placeholder("float", [None, n_steps, n_input])

    # lets have place holder for output
    y = tf.placeholder("float", [None, n_classes])

    # lets call the model
    pred = RNN(x, weights, biases)

    # We define the cost and later the optimization algo to reduce it
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    # optimizing algo to reduce it, we are using Adam
    optimizer = tf.train.AdamOptimizer\
                (learning_rate=learning_rate).minimize(cost)

    # We define the accuracy that will be displayed during the computation:
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # lets start the session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 1
        while step * batch_size < training_iters:
            # get the data out of mnist
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # reshape it
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))

            # lets run the session on optimizer
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            if step % display_step == 0:
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            step += 1

        print("Optimization Finished!")

        test_len = 128

        test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
        test_label = mnist.test.labels[:test_len]
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

if __name__ == "__main__":
    run()