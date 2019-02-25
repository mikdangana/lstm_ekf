
import numpy as np
import tensorflow as tf
sess = tf.Session()
x = tf.placeholder("float", [2, 3])
w = tf.Variable(tf.random_normal([3, 3]), name='w')
y = tf.matmul(x, w)
build = lambda fn : list(map(fn, range(0,3)))
lbl = build(lambda i: tf.placeholder("float", [2, 1]))
relu_out = tf.nn.relu(y)
lstm = tf.nn.rnn_cell.LSTMCell(1)
#x = tf.split(x, 3, 1)
print(str(x))
outputs, states = tf.contrib.rnn.static_rnn(lstm, inputs=[x], dtype=tf.float32)
cost = build(lambda i: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=lbl[i])))
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001)
train_op = build(lambda i: optimizer.minimize(cost[i]))
# tf.initialize_all_variables() is deprecated.
sess.run(tf.global_variables_initializer())
print(outputs)
for result,total_cost in map(lambda i: sess.run([train_op[i], cost[i]], feed_dict={x:np.array([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]]), lbl[i]:np.array([[1.0],[4.0]])}), range(0,3)):
    print(result)
    print(total_cost)
