from tensorflow import keras

# Define Classifier layers neuron
def create_classifier(class_count, hidden_layer, dropout_rate):
    classifier = keras.Sequential([
        keras.layers.BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
        keras.layers.Dense(hidden_layer, 
        kernel_regularizer = keras.regularizers.l2(l= 0.016), 
        activity_regularizer = keras.regularizers.l1(0.006),
        bias_regularizer = keras.regularizers.l1(0.006), 
        activation = 'relu'),
        keras.layers.Dropout(rate= dropout_rate, seed= 123),
        keras.layers.Dense(class_count, activation= 'softmax')
        ])

    return classifier
