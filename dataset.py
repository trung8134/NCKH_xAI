import tensorflow as tf

def tf_dataset(img_size, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory='Datasets/training',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        seed=1,
        shuffle=True,
        image_size=(img_size, img_size),
        color_mode='rgb',
        validation_split=0.2,
        subset="training"
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory='Datasets/training',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        seed=1,
        shuffle=True,
        image_size=(img_size, img_size),
        color_mode='rgb',
        validation_split=0.2,
        subset="validation"
    )
    
    
    return train_ds, test_ds
