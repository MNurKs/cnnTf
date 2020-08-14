import tensorflow as tf
import disable_log
from datetime import datetime
from dirs import train_dirs, validation_dirs, model_dirs
import os

"""
TODO: Use DenseNet121 427 layers
TODO: v1 -> intitial not fine tune lr=default
TODO: v2 -> fine tune, lr=1-e5, 200 layer trainable

"""
MODEL_NAME = os.path.basename(__file__).replace(".py","")
TUNNING = 100
IMAGE_SIZE = 128
CLASSES_NUM = 5
COLOR_MODE = 'rgb'
CLASSES_MODE = 'categorical'
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1) 
OPTIMIZER = tf.keras.optimizers.Adam()
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']
DROPOUT = 0.3
ACTIVATION_CONV = 'relu'
ACTIVATION_DENSE = 'relu'
ACTIVATION_DENSE_END = 'sigmoid'

def model():
    print("Create Model")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same',
                     activation=ACTIVATION_CONV, input_shape=224,224,1))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    #BLOCK1
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation=ACTIVATION_CONV))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    #BLOCK2
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=ACTIVATION_CONV))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    #BLOCK3
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=ACTIVATION_CONV))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    #CLASIFICATION
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(DROPOUT))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(DROPOUT))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(DROPOUT))
    model.add(tf.keras.layers.Dense(32, activation='softmax'))

    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=METRICS)

    return model

def beforeCompile(model):
    model.layers[0].trainable = True
    for layer in model.layers[0].layers[:227]:
        layer.trainable = False
    return model


m = model()

m.layers[0].summary()

print(f"TOTAL LAYER : {len(m.layers) + len(m.layers[0].layers) -1}")
