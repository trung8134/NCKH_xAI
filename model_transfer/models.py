import tensorflow.keras.backend as K
import tensorflow 
from tensorflow.keras import layers as L
from tensorflow import keras
from keras import layers
from keras.applications import resnet50, vgg16, EfficientNetV2S
from keras.models import Model

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

# VGGViT
def VGG16ViT_model(img_shape, class_count):
    K.clear_session()
    base_model = vgg16.VGG16(input_shape=img_shape, include_top=False, weights="imagenet")
    
    for layer in base_model.layers:
        layer.trainable = False
    
    # Get the output of block4_conv3
    block4_conv3_output = base_model.get_layer('block4_conv3').output
    
    # Attention mechanism
    query_cnn_layer = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv_attention_1')
    query_seq_encoding = query_cnn_layer(block4_conv3_output)
    
    value_cnn_layer = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv_attention_2')
    value_seq_encoding = value_cnn_layer(block4_conv3_output)
    
    query_value_attention_seq = layers.Attention()([query_seq_encoding, value_seq_encoding])
    
    # MaxPooling2D for query_value_attention_seq.
    pooling_behind_attention = tensorflow.keras.layers.MaxPooling2D(name='block4_pool')(query_value_attention_seq)
    
    block5_conv1_layer = base_model.get_layer('block5_conv1')(pooling_behind_attention)
    block5_conv2_layer = base_model.get_layer('block5_conv2')(block5_conv1_layer)
    block5_conv3_layer = base_model.get_layer('block5_conv3')(block5_conv2_layer)
    
    # MaxPooling2D for block5
    pooling_block5 = layers.MaxPooling2D(name='block5_pool')(block5_conv3_layer)
    
    # Flatten
    flatten = layers.Flatten()(pooling_block5)
    
    # output layer 
    output_layer = layers.Dense(class_count, activation='softmax')(flatten)
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=output_layer)

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
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
