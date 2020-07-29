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
IMAGE_SIZE = 224
CLASSES_NUM = 26
COLOR_MODE = 'rgb'
CLASSES_MODE = 'categorical'
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
OPTIMIZER = tf.keras.optimizers.Adam()
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']

def model():
    #? Create the base model from the pre-trained model DenseNet121
    base_model = tf.keras.applications.DenseNet121(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    print(len(base_model.layers))

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(CLASSES_NUM, activation='softmax')
    ])
    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=METRICS)

    return model

def beforeCompile(model):
    model.layers[0].trainable = True
    for layer in model.layers[0].layers[:100]:
        layer.trainable = False
    return model


# m = model()
#
# m.layers[0].summary()
#
# print(f"TOTAL LAYER : {len(m.layers) + len(m.layers[0].layers) -1}")
