from tensorflow import keras
import tensorflow as tf


data_augmentation = keras.Sequential(
        [
            keras.layers.Rescaling(1./255),
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.02), 
        ]
    )


def augment_data(inputs):
    augment = data_augmentation(inputs)
    
    return augment