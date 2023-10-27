from tensorflow import keras
from keras import layers
from keras.applications import efficientnet, mobilenet, inception_v3, resnet50, vgg16
from keras.models import Model

# EfficientNet
def EfficientNetB0_model(img_shape, class_count, hidden_layer, dropout_rate):
    input_layer = layers.Input(shape=img_shape)
    
    base_model = efficientnet.EfficientNetB0(include_top=False, weights="imagenet", pooling='max')
    x = base_model(input_layer)
    
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = layers.Dense(hidden_layer, 
                     kernel_regularizer=keras.regularizers.l2(l=0.016), 
                     activity_regularizer=keras.regularizers.l1(0.006),
                     bias_regularizer=keras.regularizers.l1(0.006), 
                     activation='relu')(x)
    x = layers.Dropout(rate=dropout_rate, seed=123)(x)
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    output_layer = layers.Dense(class_count, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# MobileNet
def MobileNetV1_model(img_shape, class_count, hidden_layer, dropout_rate):
    input_layer = layers.Input(shape=img_shape)
    
    base_model = mobilenet.MobileNet(include_top=False, weights="imagenet", pooling='max')
    x = base_model(input_layer)
    
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = layers.Dense(hidden_layer, 
                     kernel_regularizer=keras.regularizers.l2(l=0.016), 
                     activity_regularizer=keras.regularizers.l1(0.006),
                     bias_regularizer=keras.regularizers.l1(0.006), 
                     activation='relu')(x)
    x = layers.Dropout(rate=dropout_rate, seed=123)(x)
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    output_layer = layers.Dense(class_count, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Inception
def InceptionV3_model(img_shape, class_count, hidden_layer, dropout_rate):
    input_layer = layers.Input(shape=img_shape)
    
    base_model = inception_v3.InceptionV3(include_top=False, weights="imagenet", pooling='max')
    x = base_model(input_layer)
    
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = layers.Dense(hidden_layer, 
                     kernel_regularizer=keras.regularizers.l2(l=0.016), 
                     activity_regularizer=keras.regularizers.l1(0.006),
                     bias_regularizer=keras.regularizers.l1(0.006), 
                     activation='relu')(x)
    x = layers.Dropout(rate=dropout_rate, seed=123)(x)
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    output_layer = layers.Dense(class_count, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# ResNet
def ResNet50_model(img_shape, class_count, hidden_layer, dropout_rate):
    input_layer = layers.Input(shape=img_shape)
    
    base_model = resnet50.ResNet50(include_top=False, weights="imagenet", pooling='max')
    x = base_model(input_layer)
    
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = layers.Dense(hidden_layer, 
                     kernel_regularizer=keras.regularizers.l2(l=0.016), 
                     activity_regularizer=keras.regularizers.l1(0.006),
                     bias_regularizer=keras.regularizers.l1(0.006), 
                     activation='relu')(x)
    x = layers.Dropout(rate=dropout_rate, seed=123)(x)
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    output_layer = layers.Dense(class_count, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# VGG
def VGG16_model(img_shape, class_count, hidden_layer, dropout_rate):
    input_layer = layers.Input(shape=img_shape)
    
    base_model = vgg16.VGG16(include_top=False, weights="imagenet", pooling='max')
    x = base_model(input_layer)
    
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = layers.Dense(hidden_layer, 
                     kernel_regularizer=keras.regularizers.l2(l=0.016), 
                     activity_regularizer=keras.regularizers.l1(0.006),
                     bias_regularizer=keras.regularizers.l1(0.006), 
                     activation='relu')(x)
    x = layers.Dropout(rate=dropout_rate, seed=123)(x)
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    output_layer = layers.Dense(class_count, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

