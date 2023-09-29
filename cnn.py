import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Input, ZeroPadding3D, Dense, Dropout, Activation, Convolution3D, Reshape, Concatenate
from tensorflow.keras.layers import AveragePooling3D, GlobalAveragePooling3D, MaxPooling3D, LayerNormalization, BatchNormalization, ReLU
from tensorflow_addons.layers import GroupNormalization
# GroupsNormalization(groups=8)

def cnn_model(shape=(112,96,26),classes=1):

    regularizer = regularizers.l2(1e-4)    

    img_input1 = Input((shape[0],shape[1],shape[2]),name='T2W')
    xA = tf.expand_dims(img_input1,-1)
    xA = Convolution3D(filters=24, kernel_size=3, activation="relu", padding="same")(xA)
    xA = MaxPooling3D(pool_size=2)(xA)
    xA = GroupNormalization(groups=8)(xA)

    img_input2 = Input((shape[0],shape[1],shape[2]),name='ADC')
    xB = tf.expand_dims(img_input2,-1)
    xB = Convolution3D(filters=24, kernel_size=3, activation="relu", padding="same")(xB)
    xB = MaxPooling3D(pool_size=2)(xB)
    xB = GroupNormalization(groups=8)(xB)

    img_input3 = Input((shape[0],shape[1],shape[2]),name='DWI')
    xC = tf.expand_dims(img_input3,-1)
    xC = Convolution3D(filters=24, kernel_size=3, activation="relu", padding="same")(xC)
    xC = MaxPooling3D(pool_size=2)(xC)
    xC = GroupNormalization(groups=8)(xC)
        
    img_input4 = Input((shape[0],shape[1],shape[2],3),name='DCE')
    xD = Convolution3D(filters=24, kernel_size=3, activation="relu", padding="same")(img_input4)
    xD = MaxPooling3D(pool_size=2)(xD)
    xD = GroupNormalization(groups=8)(xD)

    x = Concatenate(axis=-1)([xA,xB,xC,xD])

    x = Convolution3D(filters=96, kernel_size=3, activation="relu", padding="same")(x)
    x = MaxPooling3D(pool_size=2)(x)
    x = GroupNormalization(groups=8)(x)

    x = Convolution3D(filters=192, kernel_size=3, activation="relu", padding="same")(x)
    x = MaxPooling3D(pool_size=2)(x)
    x = GroupNormalization(groups=8)(x)

    x = Convolution3D(filters=320, kernel_size=3, activation="relu", padding="same", name = "gradcam_layer")(x)
    x = MaxPooling3D(pool_size=2)(x)
    x = GroupNormalization(groups=8)(x)
    x = GlobalAveragePooling3D()(x)

    x = Dense(units=480, activation="relu")(x)
    # x = Dropout(0.3)(x)

    outputs = Dense(units=1,activation="sigmoid")(x)
    
    model = Model([img_input1,img_input2,img_input3,img_input4], outputs, name="3dcnn")
        
    return model

if __name__ == "__main__":
    model = cnn_model()
    model.summary(line_length=125)
    from tensorflow.keras.utils import plot_model
    plot_model(model, "/home/m203898/Videos/haha.png")