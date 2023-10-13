from tensorflow import keras
from model_transfer.classifier import create_classifier

# EfficientNet
def EfficientNetB0_model(img_shape, class_count, hidden_layer, dropout_rate):
    base_model=keras.applications.efficientnet.EfficientNetB0(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')

    model=keras.Sequential([
        base_model,
        create_classifier(class_count, hidden_layer, dropout_rate)
    ])

    model.compile(keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# MobileNet
def MobileNetV1_model(img_shape, class_count, hidden_layer, dropout_rate):
    base_model=keras.applications.MobileNet(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')

    model=keras.Sequential([
        base_model,
        create_classifier(class_count, hidden_layer, dropout_rate)
    ])

    model.compile(keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def MobileNetV2_model(img_shape, class_count, hidden_layer, dropout_rate):
    base_model=keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')

    model=keras.Sequential([
        base_model,
        create_classifier(class_count, hidden_layer, dropout_rate)
    ])

    model.compile(keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# ResNet
def ResNet50_model(img_shape, class_count, hidden_layer, dropout_rate):
    base_model=keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')

    model=keras.Sequential([
        base_model,
        create_classifier(class_count, hidden_layer, dropout_rate)
    ])

    model.compile(keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# VGG
def VGG16_model(img_shape, class_count, hidden_layer, dropout_rate):
    base_model=keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')

    model=keras.Sequential([
        base_model,
        create_classifier(class_count, hidden_layer, dropout_rate)
    ])

    model.compile(keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

