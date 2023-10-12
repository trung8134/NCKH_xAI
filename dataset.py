from keras.preprocessing.image import ImageDataGenerator

def tf_dataset(img_size, batch_size):
    # C:/Users/caotr/D. Computer Science/Data Science/DL/Project/NCKH-2024/
    train_path = 'Datasets/data/training'
    test_path = 'Datasets/data/testing'
    # Data generators
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

    train_batches = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    valid_batches = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    test_batches = train_datagen.flow_from_directory(
        test_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
    )
    
    return train_batches, valid_batches, test_batches
    