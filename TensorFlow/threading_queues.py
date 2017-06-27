# https://www.tensorflow.org/programmers_guide/threading_and_queues
# https://blog.metaflow.fr/tensorflow-how-to-optimise-your-input-pipeline-with-queues-and-multi-threading-e7c3874157e0

import tensorflow as tf

# Create input data with three samples from a normal distribution with mean 0 and stddev 1
input_data = tf.random_normal([3], mean=0, stddev=1)

# Create a queue that holds three elements
q = tf.FIFOQueue(3, tf.float32)

# Fill the queue with the data
input_data = tf.Print(input_data, data=[input_data], message="Raw inputs data generated:", summarize=3)
init = q.enqueue_many(input_data)

# To leverage multi-threading we create a "QueueRunner"
# that will handle the "enqueue_op" outside of the main thread
# We don't need much parallelism here, so we will use only 1 thread
numberOfThreads = 1
qr = tf.train.QueueRunner(q, [init] * numberOfThreads)
# Don't forget to add your "QueueRunner" to the QUEUE_RUNNERS collection
tf.train.add_queue_runner(qr)

# Dequeue op is used to get the next elements in the queue
x = q.dequeue()
# Each time we use the input tensor, we print the number of elements left in the queue
x = tf.Print(x, data=[q.size(), x], message="Nb elements left:")
# Iterate through the queue
y = x + 1

with tf.Session() as sess:
    # But now we build our coordinator to coordinate our child threads with
    # the main thread
    coord = tf.train.Coordinator()

    # Beware, if you don't start all your queues before runnig anything
    # The main threads will wait for them to start and you will hang again
    # This helper start all queues in tf.GraphKeys.QUEUE_RUNNERS
    threads = tf.train.start_queue_runners(coord=coord)

    # Start the queue until it's full
    sess.run(y)
    sess.run(y)
    sess.run(y)

    # Queue is full
    sess.run(y)

    # We request our child threads to stop ...
    coord.request_stop()
    # ... and we wait for them to do so before releasing the main thread
    coord.join(threads)
