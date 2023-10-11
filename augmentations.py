from tensorflow import keras
import tensorflow as tf


data_augmentation = keras.Sequential(
        [
            keras.layers.Rescaling(1./255),
        ]
    )


def augment_data(inputs):
    augment = data_augmentation(inputs)
    
    return augment