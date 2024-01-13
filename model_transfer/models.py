import tensorflow.keras.backend as K
import tensorflow 
from tensorflow.keras import layers as L
from tensorflow import keras
from keras import layers 
from keras.applications import resnet50, vgg16, EfficientNetV2S
from keras.models import Model
<<<<<<< HEAD
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam

=======
>>>>>>> 8be988906922aa09be900eda7e09b27b0b27b256

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
<<<<<<< HEAD
    def attach_attention_module(net, attention_module):
        if attention_module == 'cbam_block': # CBAM_block
            net = cbam_block(net)
        else:
            raise Exception("'{}' is not supported attention module!".format(attention_module))

        return net

    def cbam_block(cbam_feature, ratio=8):
        """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
        As described in https://arxiv.org/abs/1807.06521.
        """
        
        cbam_feature = channel_attention(cbam_feature, ratio)
        cbam_feature = spatial_attention(cbam_feature)
        return cbam_feature

    def channel_attention(input_feature, ratio=8):
        
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        channel = input_feature.shape[channel_axis]
        
        shared_layer_one = Dense(channel//ratio,
                                activation='relu',
                                kernel_initializer='he_normal',
                                use_bias=True,
                                bias_initializer='zeros')
        shared_layer_two = Dense(channel,
                                kernel_initializer='he_normal',
                                use_bias=True,
                                bias_initializer='zeros')
        
        avg_pool = GlobalAveragePooling2D()(input_feature)    
        avg_pool = Reshape((1,1,channel))(avg_pool)
        assert avg_pool.shape[1:] == (1,1,channel)
        avg_pool = shared_layer_one(avg_pool)
        assert avg_pool.shape[1:] == (1,1,channel//ratio)
        avg_pool = shared_layer_two(avg_pool)
        assert avg_pool.shape[1:] == (1,1,channel)
        
        max_pool = GlobalMaxPooling2D()(input_feature)
        max_pool = Reshape((1,1,channel))(max_pool)
        assert max_pool.shape[1:] == (1,1,channel)
        max_pool = shared_layer_one(max_pool)
        assert max_pool.shape[1:] == (1,1,channel//ratio)
        max_pool = shared_layer_two(max_pool)
        assert max_pool.shape[1:] == (1,1,channel)
        
        cbam_feature = Add()([avg_pool,max_pool])
        cbam_feature = Activation('sigmoid')(cbam_feature)
        
        if K.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)
        
        return multiply([input_feature, cbam_feature])

    def spatial_attention(input_feature):
        kernel_size = 7
        
        if K.image_data_format() == "channels_first":
            channel = input_feature.shape[1]
            cbam_feature = Permute((2,3,1))(input_feature)
        else:
            channel = input_feature.shape[-1]
            cbam_feature = input_feature
        
        avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
        assert avg_pool.shape[-1] == 1
        max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
        assert max_pool.shape[-1] == 1
        concat = Concatenate(axis=3)([avg_pool, max_pool])
        assert concat.shape[-1] == 2
        cbam_feature = Conv2D(filters = 1,
                        kernel_size=kernel_size,
                        strides=1,
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer='he_normal',
                        use_bias=False)(concat)	
        assert cbam_feature.shape[-1] == 1
        
        if K.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)
            
        return multiply([input_feature, cbam_feature])

    def vgg16_cbam(input_shape, num_classes=11, attention_module=None):
        """VGG16 with CBAM Model builder

        # Arguments
            input_shape (tensor): shape of input image tensor
            num_classes (int): number of classes

        # Returns
            model (Model): Keras model instance
        """
        base_model = vgg16(weights='imagenet', include_top=False, input_shape=input_shape)

        # Freeze the convolutional layers
        for layer in base_model.layers:
            layer.trainable = False

        # VGG16 Block 1
        x = base_model.get_layer('block1_conv1').output
        x = base_model.get_layer('block1_conv2')(x) 
        if attention_module is not None:
            x = attach_attention_module(x, attention_module)
        x = MaxPooling2D(name='block1_pool')(x)

        # VGG16 Block 2
        x = base_model.get_layer('block2_conv1')(x)
        x = base_model.get_layer('block2_conv2')(x) 
        if attention_module is not None:
            x = attach_attention_module(x, attention_module)
        x = MaxPooling2D(name='block2_pool')(x)

        # VGG16 Block 3
        x = base_model.get_layer('block3_conv1')(x)
        x = base_model.get_layer('block3_conv2')(x)
        x = base_model.get_layer('block3_conv3')(x) 
        if attention_module is not None:
            x = attach_attention_module(x, attention_module)
        x = MaxPooling2D(name='block3_pool')(x)

        # VGG16 Block 4
        x = base_model.get_layer('block4_conv1')(x)
        x = base_model.get_layer('block4_conv2')(x)
        x = base_model.get_layer('block4_conv3')(x) 
        if attention_module is not None:
            x = attach_attention_module(x, attention_module)
        x = MaxPooling2D(name='block4_pool')(x)

        # VGG16 Block 5
        x = base_model.get_layer('block5_conv1')(x)
        x = base_model.get_layer('block5_conv2')(x)
        x = base_model.get_layer('block5_conv3')(x) 
        if attention_module is not None:
            x = attach_attention_module(x, attention_module)
        x = MaxPooling2D(name='block5_pool')(x)

        # Classification head
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        outputs = Dense(num_classes, activation='softmax')(x)

        # Instantiate model.
        model = Model(inputs=base_model.input, outputs=outputs)
        return model

    model = vgg16_cbam(img_shape, class_count, 'cbam_block')

    model.compile(loss='categorical_crossentropy',
            optimizer=Adam(lr=1e-3),
            metrics=['accuracy'])

=======
    base_model = vgg16.VGG16(input_shape=img_shape, include_top=False, weights="imagenet")
    
    for layer in base_model.layers:
        layer.trainable = False
    
    # Get the output of block4_conv3
    block1_conv2_output = base_model.get_layer('block1_conv2').output
    
    # Attention mechanism
    query_cnn_layer = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='conv_attention_1')
    query_seq_encoding = query_cnn_layer(block1_conv2_output)
    
    value_cnn_layer = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='conv_attention_2')
    value_seq_encoding = value_cnn_layer(block1_conv2_output)
    
    query_value_attention_seq = layers.Attention()([query_seq_encoding, value_seq_encoding])
    
    # MaxPooling2D for query_value_attention_seq.
    pooling_behind_attention = layers.MaxPooling2D(name='block1_pool')(query_value_attention_seq)
    
    # block2
    block2_conv1_layer = base_model.get_layer('block2_conv1')(pooling_behind_attention)
    block2_conv2_layer = base_model.get_layer('block2_conv2')(block2_conv1_layer)
    pooling2 = layers.MaxPooling2D(name='block2_pool')(block2_conv2_layer)
    
    # block3
    block3_conv1_layer = base_model.get_layer('block3_conv1')(pooling2)
    block3_conv2_layer = base_model.get_layer('block3_conv2')(block3_conv1_layer)
    block3_conv3_layer = base_model.get_layer('block3_conv3')(block3_conv2_layer)
    pooling3 = layers.MaxPooling2D(name='block3_pool')(block3_conv3_layer)
    
    # block4
    block4_conv1_layer = base_model.get_layer('block4_conv1')(pooling3)
    block4_conv2_layer = base_model.get_layer('block4_conv2')(block4_conv1_layer)
    block4_conv3_layer = base_model.get_layer('block4_conv3')(block4_conv2_layer)
    pooling4 = layers.MaxPooling2D(name='block4_pool')(block4_conv3_layer)
    
    # block5
    block5_conv1_layer = base_model.get_layer('block5_conv1')(pooling4)
    block5_conv2_layer = base_model.get_layer('block5_conv2')(block5_conv1_layer)
    block5_conv3_layer = base_model.get_layer('block5_conv3')(block5_conv2_layer)
    pooling5 = layers.MaxPooling2D(name='block5_pool')(block5_conv3_layer)
    
    # Flatten
    flatten = layers.Flatten()(pooling5)
    
    # output layer 
    output_layer = layers.Dense(11, activation='softmax')(flatten)
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=output_layer)
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
>>>>>>> 8be988906922aa09be900eda7e09b27b0b27b256
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
