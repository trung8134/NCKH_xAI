from tensorflow import keras
from keras import layers
from keras.applications import efficientnet, mobilenet, inception_v3, resnet50, vgg16
from keras.models import Model

# EfficientNet
def EfficientNetB0_model(img_shape, class_count, hidden_layer, dropout_rate):
    base_model = efficientnet.EfficientNetB0(input_shape=img_shape, include_top=False, weights="imagenet")
    
    for layer in base_model.layers:
        layer.trainable = False
    
    last_layer = base_model.get_layer('top_conv')
    last_output = last_layer.output
    
    x = layers.Flatten()(last_output)
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = layers.Dense(hidden_layer, 
                     kernel_regularizer=keras.regularizers.l2(l=0.016), 
                     activity_regularizer=keras.regularizers.l1(0.006),
                     bias_regularizer=keras.regularizers.l1(0.006), 
                     activation='relu')(x)
    x = layers.Dropout(rate=dropout_rate, seed=123)(x)
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = layers.Dense(class_count, activation='softmax')(x)
    
    model = Model(base_model.input, x) 

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# MobileNet
def MobileNetV1_model(img_shape, class_count, hidden_layer, dropout_rate):
    base_model = mobilenet.MobileNet(input_shape=img_shape, include_top=False, weights="imagenet")
    
    for layer in base_model.layers:
        layer.trainable = False
    
    last_layer = base_model.get_layer('conv_pw_13')
    last_output = last_layer.output
    
    x = layers.Flatten()(last_output)
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = layers.Dense(hidden_layer, 
                     kernel_regularizer=keras.regularizers.l2(l=0.016), 
                     activity_regularizer=keras.regularizers.l1(0.006),
                     bias_regularizer=keras.regularizers.l1(0.006), 
                     activation='relu')(x)
    x = layers.Dropout(rate=dropout_rate, seed=123)(x)
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = layers.Dense(class_count, activation='softmax')(x)
    
    model = Model(base_model.input, x) 
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Inception
def InceptionV3_model(img_shape, class_count, hidden_layer, dropout_rate):
    base_model = inception_v3.InceptionV3(input_shape=img_shape, include_top=False, weights="imagenet")
    
    for layer in base_model.layers:
        layer.trainable = False
    
    last_layer = base_model.get_layer('mixed7')
    last_output = last_layer.output
    
    x = layers.Flatten()(last_output)
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = layers.Dense(hidden_layer, 
                     kernel_regularizer=keras.regularizers.l2(l=0.016), 
                     activity_regularizer=keras.regularizers.l1(0.006),
                     bias_regularizer=keras.regularizers.l1(0.006), 
                     activation='relu')(x)
    x = layers.Dropout(rate=dropout_rate, seed=123)(x)
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = layers.Dense(class_count, activation='softmax')(x)
    
    model = Model(base_model.input, x) 
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# ResNet
def ResNet50_model(img_shape, class_count, hidden_layer, dropout_rate):
    base_model = resnet50.ResNet50(input_shape=img_shape, include_top=False, weights="imagenet")
    
    for layer in base_model.layers:
        layer.trainable = False
    
    last_layer = base_model.get_layer('conv5_block3_3_conv')
    last_output = last_layer.output
    
    x = layers.Flatten()(last_output)
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = layers.Dense(hidden_layer, 
                     kernel_regularizer=keras.regularizers.l2(l=0.016), 
                     activity_regularizer=keras.regularizers.l1(0.006),
                     bias_regularizer=keras.regularizers.l1(0.006), 
                     activation='relu')(x)
    x = layers.Dropout(rate=dropout_rate, seed=123)(x)
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = layers.Dense(class_count, activation='softmax')(x)
    
    model = Model(base_model.input, x) 
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# VGG
def VGG16_model(img_shape, class_count, hidden_layer, dropout_rate):
    base_model = vgg16.VGG16(input_shape=img_shape, include_top=False, weights="imagenet")
    
    for layer in base_model.layers:
        layer.trainable = False
    
    last_layer = base_model.get_layer('block5_conv3')
    last_output = last_layer.output
    
    x = layers.Flatten()(last_output)
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = layers.Dense(hidden_layer, 
                     kernel_regularizer=keras.regularizers.l2(l=0.016), 
                     activity_regularizer=keras.regularizers.l1(0.006),
                     bias_regularizer=keras.regularizers.l1(0.006), 
                     activation='relu')(x)
    x = layers.Dropout(rate=dropout_rate, seed=123)(x)
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = layers.Dense(class_count, activation='softmax')(x)
    
    model = Model(base_model.input, x) 
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

