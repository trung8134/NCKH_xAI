import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow import keras
from keras import layers
from keras.applications import resnet50, vgg16, EfficientNetV2S
from keras.models import Model
from model_transfer.config import MobileViT 

def MobileViT_S(input_shape, class_count):
    K.clear_session()
    num_channels = [16, 32, 64, 64, 64, 96, 144, 128, 192, 160, 240, 640]
    dim = [144, 192, 240]
    expansion_ratio = 4

    model = MobileViT(
        input_shape,
        num_channels,
        dim,
        expansion_ratio,
        num_classes=class_count
    )
    
    last_layer = model.get_layer('conv2d_27')
    last_output = last_layer.output
    
    x = layers.BatchNormalization()(last_output)
    layers.Activation('swish')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(class_count, activation='softmax')(x)
    
    model = Model(model.input, x)
    
    optimizer = keras.optimizers.AdamW(weight_decay=0.01, learning_rate=0.002)
    loss_function = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    
    return model

def MobileViT_XS(input_shape, class_count):
    K.clear_session()
    num_channels = [16, 32, 48, 48, 48, 64, 96, 80, 120, 96, 144, 384]
    dim = [96, 120, 144]
    expansion_ratio = 4


    model = MobileViT(
        input_shape,
        num_channels,
        dim,
        expansion_ratio,
        num_classes=class_count
    )
    
    last_layer = model.get_layer('conv2d_27')
    last_output = last_layer.output
    
    x = layers.BatchNormalization()(last_output)
    layers.Activation('swish')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(class_count, activation='softmax')(x)
    
    model = Model(model.input, x)
    
    optimizer = keras.optimizers.AdamW(weight_decay=0.01, learning_rate=0.002)
    loss_function = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    
    return model

def MobileViT_XXS(input_shape, class_count):
    K.clear_session()
    num_channels = [16, 16, 24, 24, 24, 48, 64, 64, 80, 80, 96, 320]
    dim = [64, 80, 96]
    expansion_ratio = 2

    model = MobileViT(
        input_shape,
        num_channels,
        dim,
        expansion_ratio,
        num_classes=class_count
    )
    
    last_layer = model.get_layer('conv2d_27')
    last_output = last_layer.output
    
    x = layers.BatchNormalization()(last_output)
    layers.Activation('swish')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(class_count, activation='softmax')(x)
    
    model = Model(model.input, x)
    
    optimizer = keras.optimizers.AdamW(weight_decay=0.01, learning_rate=0.002)
    loss_function = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    
    return model
    

# ResNet
def ResNet50_model(img_shape, class_count):
    base_model = resnet50.ResNet50(input_shape=img_shape, include_top=False, weights="imagenet")
    
    for layer in base_model.layers:
        layer.trainable = False
    
    last_layer = base_model.get_layer('conv5_block3_3_conv')
    last_output = last_layer.output
    
    x = layers.GlobalAveragePooling2D()(last_output)
    x = layers.Dense(class_count, activation='softmax')(x)
    
    model = Model(base_model.input, x) 
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# VGG
def VGG16_model(img_shape, class_count):
    base_model = vgg16.VGG16(input_shape=img_shape, include_top=False, weights="imagenet")
    
    for layer in base_model.layers:
        layer.trainable = False
    
    last_layer = base_model.get_layer('block5_conv3')
    last_output = last_layer.output
    
    x = layers.GlobalAveragePooling2D()(last_output)
    x = layers.Flatten()(x)
    x = layers.Dense(class_count, activation='softmax')(x)
    
    model = Model(base_model.input, x) 
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# EfficientNetV2S
def EfficientNetV2S_model(img_shape, class_count):
    base_model = EfficientNetV2S(input_shape=img_shape, include_top=False, weights="imagenet")
    
    for layer in base_model.layers:
        layer.trainable = False
    
    last_layer = base_model.get_layer('top_conv')
    last_output = last_layer.output
    
    x = layers.GlobalAveragePooling2D()(last_output)
    x = layers.Flatten()(x)
    x = layers.Dense(class_count, activation='softmax')(x)
    
    model = Model(base_model.input, x) 
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
