from keras.preprocessing.image import ImageDataGenerator

def tf_dataset(img_size, batch_size):
    dataset_path = 'Datasets/training'

    # Data generators
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_batches = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    valid_batches = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    test_batches = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        subset='validation'
    )
    
    return train_batches, valid_batches, test_batches
    