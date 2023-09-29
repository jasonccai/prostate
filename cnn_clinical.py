import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Input, ZeroPadding3D, Dense, Dropout, Activation, Convolution3D, Reshape, Concatenate
from tensorflow.keras.layers import AveragePooling3D, GlobalAveragePooling3D, MaxPooling3D, LayerNormalization, BatchNormalization, ReLU
from tensorflow_addons.layers import GroupNormalization
# GroupsNormalization(groups=4)

def dnn_model():
    inputs = Input(shape=(3,))
    X = Dense(units=12,activation="relu")(inputs)
    X = Dense(units=6,activation="relu")(X)
    # X = Dense(units=3,activation="relu")(X)
    output = Dense(units=1,activation="sigmoid")(X)    
    model = Model(inputs, output, name="DNN")
    return model

if __name__ == "__main__":
    model = dnn_model()
    model.summary()
    from tensorflow.keras.utils import plot_model
    plot_model(model, "/home/m203898/Videos/haha.png")