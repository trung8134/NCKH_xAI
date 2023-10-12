from tensorflow import keras
from classification_models.tfkeras import Classifiers

def create_encoder_ResNet18(img_size, num_classes):
    # Load the resnet50 model with pretrained weights
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    resnet = ResNet18(
            include_top = False, 
            pooling = 'avg',
            classes = num_classes,
            weights = 'imagenet', 
            input_shape = (img_size, img_size, 3), 
            # pooling="avg", # Vector đặc trưng 1 chiều(có thể thay thế cho lớp Flatten)
        )

    inputs = keras.Input(shape=(img_size, img_size, 3))
    outputs = resnet(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name="Encoder-ResNet18")
    
    return model

def create_encoder_ResNet50(img_size, num_classes):
    # Load the resnet50 model with pretrained weights
    resnet = keras.applications.ResNet50(
            include_top = False, 
            pooling = 'avg',
            classes = num_classes,
            weights='imagenet', 
            input_shape = (img_size, img_size, 3), 
            # pooling="avg", # Vector đặc trưng 1 chiều(có thể thay thế cho lớp Flatten)
        )

    inputs = keras.Input(shape=(img_size, img_size, 3))
    outputs = resnet(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name="Encoder-ResNet50")
    
    return model

