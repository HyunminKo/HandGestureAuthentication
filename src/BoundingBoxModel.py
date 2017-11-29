import tensorflow as tf
import scipy

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

def maxPool2d(x,stride):
    return tf.nn.max_pool(x, strides=[1,stride,stride,1],ksize=[1,3,3,1],padding='VALID')

x = tf.placeholder(tf.float32, shape=[None, 100, 176, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 4])

x_image = x

#first convolutional layer
W_conv1 = weight_variable([5, 5, 3, 24])
b_conv1 = bias_variable([24])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 2) + b_conv1)
p_conv1 = maxPool2d(h_conv1,2)

#second convolutional layer
W_conv2 = weight_variable([5, 5, 24, 36])
b_conv2 = bias_variable([36])

h_conv2 = tf.nn.relu(conv2d(p_conv1, W_conv2, 2) + b_conv2)

#third convolutional layer
W_conv3 = weight_variable([5, 5, 36, 48])
b_conv3 = bias_variable([48])

h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 2) + b_conv3)
p_conv3 = maxPool2d(h_conv3,2)

#FCL 1
W_fc1 = weight_variable([144, 288])
b_fc1 = bias_variable([288])

p_conv3_flat = tf.reshape(p_conv3, [-1, 144])
h_fc1 = tf.nn.relu(tf.matmul(p_conv3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#FCL 2
W_fc2 = weight_variable([288, 100])
b_fc2 = bias_variable([100])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

#FCL 3
W_fc3 = weight_variable([100, 4])
b_fc3 = bias_variable([4])

#Output
y = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

