import tensorflow as tf
import numpy as np
import preprocess
import augmentation
from tqdm import tqdm
import os

#Initialise Hyperparameters
batch_size=32
train_paths, train_l, train_i, test_paths, test_l, test_i, valid_paths, valid_l, valid_i = preprocess.get_data_paths_labels()

total_train_images=len(train_paths)
total_valid_images=len(valid_paths)
total_test_images=len(test_paths)

save_dir="models/model1"
log_dir="log/aug/1"
save_model_dir='model1'

if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)

if not os.path.exists(log_dir):
      os.makedirs(log_dir)

#Define Model
x=tf.placeholder(tf.float32, shape=[None, 137, 115, 3], name='x')
x_image=tf.reshape(x, shape=[-1, 137, 115, 3])

ird=tf.placeholder(tf.float32, shape=[None, 1], name='ird')

y=tf.placeholder(tf.float32, shape=[None, 1], name='y')

y_cls=tf.argmax(y, axis=1)

conv_layer_1=tf.layers.conv2d(inputs=x_image, name="conv1", data_format='channels_last', filters=64, kernel_size=3, activation=tf.nn.relu)

maxpool1=tf.layers.max_pooling2d(inputs=conv_layer_1, pool_size=2, strides=2)

conv_layer_2=tf.layers.conv2d(inputs=maxpool1, data_format='channels_last', name="conv2", filters=32, kernel_size=3, activation=tf.nn.relu)

maxpool2=tf.layers.max_pooling2d(inputs=conv_layer_2, pool_size=2, strides=2)

conv_layer_3=tf.layers.conv2d(inputs=maxpool2, data_format='channels_last', name="conv3", filters=16, kernel_size=3, activation=tf.nn.relu)

maxpool3=tf.layers.max_pooling2d(inputs=conv_layer_3, pool_size=2, strides=2)

flatten=tf.layers.flatten(maxpool3)
flatten=tf.concat([flatten, ird], axis=-1)

dense1=tf.layers.dense(inputs=flatten, units=512, activation=tf.nn.relu)
dense2=tf.layers.dense(inputs=dense1, units=128, activation=tf.nn.relu)
dense3=tf.layers.dense(inputs=dense2, units=32, activation=tf.nn.relu)

logits=tf.layers.dense(inputs=dense3, units=1)

cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)

loss=tf.losses.mean_squared_error(y, logits)

tf.summary.scalar("Loss", loss)

optimiser=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

tf.summary.scalar("Batch MSE", loss)

session=tf.Session()

session.run(tf.global_variables_initializer())

saver=tf.train.Saver()
vis_writer=tf.summary.FileWriter(log_dir, session.graph)
merge=tf.summary.merge_all()

prev_valid_accuracy=0
current_valid_accuracy=0
ctr=0
counter=0

for j in range(200):
    print("Training Epoch", (j+1))
    acc=0
    for i in tqdm(range(int(total_train_images/batch_size)+1)):

        data_batch=train_paths[i*batch_size:(i*batch_size)+batch_size]
        x_batch=np.asarray(preprocess.reshape_data(data_batch))
        y_true_batch=np.asarray(train_l[i*batch_size:(i*batch_size)+batch_size])
        y_true_batch=np.reshape(y_true_batch, (-1, 1))
        x_i_batch=np.asarray(train_i[i*batch_size:(i*batch_size)+batch_size])
        x_i_batch=np.reshape(x_i_batch, (-1, 1))

        feed_dict_train={x: x_batch,
                        y: y_true_batch,
                        ird: x_i_batch}
        session.run(optimiser, feed_dict=feed_dict_train)
        summary, batch_loss=session.run([merge, loss], feed_dict=feed_dict_train)

        vis_writer.add_summary(summary, counter)
        acc=acc+batch_loss

    print("Train MSE for Epoch {} is = {}".format((j+1), (acc/(total_train_images/batch_size))))

    print("Validation Epoch", (j+1))
    valid_acc=0

    for i in tqdm(range(int(total_valid_images/batch_size)+1)):

        valid_data_batch=valid_paths[i*batch_size:(i*batch_size)+batch_size]
        valid_x_batch=np.asarray(preprocess.reshape_data(valid_data_batch))
        valid_y_true_batch=np.asarray(valid_l[i*batch_size:(i*batch_size)+batch_size])
        valid_y_true_batch=np.reshape(valid_y_true_batch, (-1, 1))
        valid_x_i_batch=np.asarray(valid_i[i*batch_size:(i*batch_size)+batch_size])
        valid_x_i_batch=np.reshape(valid_x_i_batch, (-1, 1))

        valid_feed_dict={x: valid_x_batch,
                            y: valid_y_true_batch,
                            ird: valid_x_i_batch}
        
        valid_batch_acc=session.run(loss, feed_dict=valid_feed_dict)
    
        valid_acc=valid_acc+valid_batch_acc

    current_valid_accuracy=(valid_acc/((total_valid_images)/batch_size))

    print("Validation MSE is = {}".format(current_valid_accuracy))

    if prev_valid_accuracy>current_valid_accuracy:
        print("Previous Validation Accurcay Greater than Current")
        ctr=ctr+1
    if ctr>2:
        break
    
    model_path=os.path.join(save_dir, ("model"+str(j)))
    saver.save(session, save_path=model_path)

    prev_valid_accuracy=valid_acc/(total_valid_images/batch_size)

test_acc=0

for i in tqdm(range(int(total_test_images/batch_size)+1)):
    test_data_batch=test_paths[i*batch_size:(i*batch_size)+batch_size]
    test_x_batch=np.asarray(preprocess.reshape_data(test_data_batch))
    test_y_true_batch=np.asarray(test_l[i*batch_size:(i*batch_size)+batch_size])
    test_y_true_batch=np.reshape(test_y_true_batch, (-1, 1))
    test_x_i_batch=np.asarray(test_i[i*batch_size:(i*batch_size)+batch_size])
    test_x_i_batch=np.reshape(test_x_i_batch, (-1, 1))

    test_feed_dict={x: test_x_batch,
                        y: test_y_true_batch,
                        ird: test_x_i_batch}
    
    test_batch_acc=session.run(loss, feed_dict=test_feed_dict)

    test_acc=test_acc+test_batch_acc

print("Test MSE= {}".format(test_acc/(total_test_images/batch_size)))

session.close()
