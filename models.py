from tensorflow import keras
from augmentations import augment_data
from vit_keras import vit
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
    augment = augment_data(inputs)
    outputs = resnet(augment)
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
    augment = augment_data(inputs)
    outputs = resnet(augment)
    model = keras.Model(inputs=inputs, outputs=outputs, name="Encoder-ResNet50")
    
    return model


def create_encoder_ViTB16(img_size, num_classes):
    vit_model = vit.vit_b16(
        image_size = img_size, # Kích thước ảnh đầu vào
        activation = 'softmax',
        pretrained = True,
        include_top = False, # Thêm lớp MLP sau lớp Transformer
        pretrained_top = False, # Không sử dụng weights đã train trên tập data gốc lên lớp MLP sau lớp Transformer
        classes = num_classes
    )

    inputs = keras.Input(shape=(img_size, img_size, 3))
    augment = augment_data(inputs)
    outputs = vit_model(augment)
    model = keras.Model(inputs=inputs, outputs=outputs, name="Encoder-ViTB16")
    
    return model

