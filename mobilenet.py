import preprocess
import keras
import tensorflow
import numpy as np

from keras.applications.mobilenet import preprocess_input, MobileNet
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.preprocessing import image
from keras.optimizers import SGD

def reshape_data(paths):
    images = []
    for path in paths:
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        #x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        images.append(x)

    return np.asarray(images)

def data_gen(paths, ird, loss, batch_size, base_model):
    while True:
        for i in range(int(len(paths)/batch_size)+1):
            data_batch=paths[i*batch_size:(i*batch_size)+batch_size]
            x_batch=np.asarray(reshape_data(data_batch))
            y_true_batch=np.asarray(loss[i*batch_size:(i*batch_size)+batch_size])
            y_true_batch=np.reshape(y_true_batch, (-1, 1))
            x_i_batch=np.asarray(ird[i*batch_size:(i*batch_size)+batch_size])
            x_i_batch=np.reshape(x_i_batch, (-1, 1))

            feed_dict=({'input_1': x_batch,
                    'ird':x_i_batch},{'y': y_true_batch}) 

            yield feed_dict

def add_new_last_layer(base_model):
    """Add last layer to the convnet
    Args:
        base_model: keras model excluding top
        nb_classes: # of classes
    Returns:
        new keras model with last layer
    """
    ird_input=Input(shape=(1,), name='ird')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x=keras.layers.concatenate([x, ird_input])
    x = Dense(128, activation='relu')(x)  #new FC layer, random init
    x = Dense(32, activation='relu')(x)  #new FC layer, random init
    predictions = Dense(1, activation='linear', name='y')(x)  #Last Prediction layer
    model = Model(inputs=[base_model.input, ird_input], outputs=predictions)
    return model

train_paths, train_l, train_i, test_paths, test_l, test_i, valid_paths, valid_l, valid_i = preprocess.get_data_paths_labels()

base_model = MobileNet(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
model = add_new_last_layer(base_model)


for layer in base_model.layers:
    layer.trainable=False

import ipdb; ipdb.set_trace()
model.compile(optimizer='adam',
        loss='mean_squared_error')

history=model.fit_generator(data_gen(train_paths, train_i, train_l, 32, base_model), steps_per_epoch=len(train_paths)/32, epochs=5,validation_data=data_gen(valid_paths, valid_i, valid_l, 32, base_model), validation_steps=len(valid_paths)/32)
