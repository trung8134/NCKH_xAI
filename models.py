from tensorflow import keras
from classifier import create_classifier

def EfficientNetB3_model(img_shape, class_count, hidden_layer, dropout_rate):
    base_model = keras.applications.efficientnet.EfficientNetB3(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')

    model = keras.Sequential([
        base_model,
        create_classifier(class_count, hidden_layer, dropout_rate)
    ])

    model.compile(keras.optimizers.Adam(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])
    
    return model

def ResNet50_model(img_shape, class_count, hidden_layer, dropout_rate):
    base_model = keras.applications.ResNet50(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')

    model = keras.Sequential([
        base_model,
        create_classifier(class_count, hidden_layer, dropout_rate)
    ])

    model.compile(keras.optimizers.Adam(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])
    
    return model