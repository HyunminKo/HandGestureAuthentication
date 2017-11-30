import os
import tensorflow as tf
import VIVA
import BoundingBoxModel

LOGDIR = './save'

L2NormConst = 0.001
epochs = 20
batch_size = 100

sess = tf.InteractiveSession()

train_vars = tf.trainable_variables()
loss = tf.reduce_mean(tf.square(tf.subtract(BoundingBoxModel.y_, BoundingBoxModel.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.global_variables_initializer())

# create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver(write_version = tf.train.SaverDef.V2)

# op to write logs to Tensorboard
logs_path = './logs'
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

# train over the dataset about 30 times
for epoch in range(epochs):
    for i in range(int(VIVA.num_images/batch_size)):
        xs, ys = VIVA.LoadTrainBatch(batch_size)
        train_step.run(feed_dict={BoundingBoxModel.x: xs, BoundingBoxModel.y_: ys, BoundingBoxModel.keep_prob: 0.8})
        loss_value = loss.eval(feed_dict={BoundingBoxModel.x:xs, BoundingBoxModel.y_: ys, BoundingBoxModel.keep_prob: 1.0})
        print("Train_Loss: %g" % (loss_value))
        if i % 10 == 0:
            xs, ys = VIVA.LoadValBatch(batch_size)
            loss_value = loss.eval(feed_dict={BoundingBoxModel.x:xs, BoundingBoxModel.y_: ys, BoundingBoxModel.keep_prob: 1.0})
            print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

        # write logs at every iteration
        summary = merged_summary_op.eval(feed_dict={BoundingBoxModel.x:xs, BoundingBoxModel.y_: ys, BoundingBoxModel.keep_prob: 1.0})
        summary_writer.add_summary(summary, epoch * VIVA.num_images/batch_size + i)

        if i % batch_size == 0:
            if not os.path.exists(LOGDIR):
                os.makedirs(LOGDIR)
            checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
            filename = saver.save(sess, checkpoint_path)
            print("Model saved in file: %s" % filename)

print("Run the command line:\n" \
          "--> tensorboard --logdir=./logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")