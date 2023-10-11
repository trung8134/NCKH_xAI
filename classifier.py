from tensorflow import keras
from keras import layers

# Định nghĩa lớp phân loại neuron
def create_classifier(encoder, input_shape, hidden_units, activation, dropout_rate, num_classes, learning_rate, trainable=True):

    # Đóng băng các layers của encoder với trainable = False
    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    features = layers.Flatten()(features)
    features = layers.Dense(hidden_units, activation = activation)(features)
    features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(hidden_units, activation = activation)(features)
    features = layers.Dropout(dropout_rate)(features)
    outputs = layers.Dense(num_classes, activation="softmax")(features)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Classifier")
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate = learning_rate),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )
    return model
