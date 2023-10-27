from tensorflow import keras

# EfficientNet
def EfficientNetB0_model(img_shape, class_count, hidden_layer, dropout_rate):
    base_model=keras.applications.efficientnet.EfficientNetB0(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')

    model=keras.Sequential([
        base_model,
        keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        keras.layers.Dense(hidden_layer, 
                        kernel_regularizer =keras.regularizers.l2(l=0.016), 
                        activity_regularizer =keras.regularizers.l1(0.006),
                        bias_regularizer =keras.regularizers.l1(0.006), 
                        activation ='relu'),
        keras.layers.Dropout(rate=dropout_rate, seed=123),
        keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        keras.layers.Dense(class_count, activation='softmax')
    ])

    model.compile(keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# MobileNet
def MobileNetV1_model(img_shape, class_count, hidden_layer, dropout_rate):
    base_model=keras.applications.MobileNet(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')

    model=keras.Sequential([
        base_model,
        keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        keras.layers.Dense(hidden_layer, 
                        kernel_regularizer =keras.regularizers.l2(l=0.016), 
                        activity_regularizer =keras.regularizers.l1(0.006),
                        bias_regularizer =keras.regularizers.l1(0.006), 
                        activation ='relu'),
        keras.layers.Dropout(rate=dropout_rate, seed=123),
        keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        keras.layers.Dense(class_count, activation='softmax')
    ])

    model.compile(keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Inception
def InceptionV3_model(img_shape, class_count, hidden_layer, dropout_rate):
    base_model=keras.applications.InceptionV3(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')

    model=keras.Sequential([
        base_model,
        keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        keras.layers.Dense(hidden_layer, 
                        kernel_regularizer =keras.regularizers.l2(l=0.016), 
                        activity_regularizer =keras.regularizers.l1(0.006),
                        bias_regularizer =keras.regularizers.l1(0.006), 
                        activation ='relu'),
        keras.layers.Dropout(rate=dropout_rate, seed=123),
        keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        keras.layers.Dense(class_count, activation='softmax')
    ])

    model.compile(keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# ResNet
def ResNet50_model(img_shape, class_count, hidden_layer, dropout_rate):
    base_model=keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')

    model=keras.Sequential([
        base_model,
        keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        keras.layers.Dense(hidden_layer, 
                        kernel_regularizer =keras.regularizers.l2(l=0.016), 
                        activity_regularizer =keras.regularizers.l1(0.006),
                        bias_regularizer =keras.regularizers.l1(0.006), 
                        activation ='relu'),
        keras.layers.Dropout(rate=dropout_rate, seed=123),
        keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        keras.layers.Dense(class_count, activation='softmax')
    ])

    model.compile(keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# VGG
def VGG16_model(img_shape, class_count, hidden_layer, dropout_rate):
    base_model=keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')

    model=keras.Sequential([
        base_model,
        keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        keras.layers.Dense(hidden_layer, 
                        kernel_regularizer =keras.regularizers.l2(l=0.016), 
                        activity_regularizer =keras.regularizers.l1(0.006),
                        bias_regularizer =keras.regularizers.l1(0.006), 
                        activation ='relu'),
        keras.layers.Dropout(rate=dropout_rate, seed=123),
        keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        keras.layers.Dense(class_count, activation='softmax')
    ])

    model.compile(keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

