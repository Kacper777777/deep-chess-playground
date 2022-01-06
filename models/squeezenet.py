import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Convolution2D, Conv2D, MaxPooling2D, AveragePooling2D, \
    GlobalAveragePooling2D, Activation, ReLU, concatenate, Concatenate, UpSampling2D, \
    Dropout, BatchNormalization, Lambda, Flatten, Dense
from tensorflow.keras.models import Model

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"


# Modular function for Fire Node.
def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x


def squeezenet(image_shape=(224, 224, 3),
               use_bn_on_input=False,
               first_stride=2,
               output_nodes=1000,
               name='squeezenet'):
    raw_image_input = tf.keras.Input(shape=image_shape)
    if use_bn_on_input:
        image_input = BatchNormalization()(raw_image_input)
    else:
        image_input = raw_image_input

    x = Convolution2D(64, (3, 3), strides=(first_stride, first_stride), padding='same', name='conv1')(image_input)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    x = Dropout(0.5, name='drop9')(x)

    x = Convolution2D(output_nodes, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)

    model = Model(image_input, out, name=name)
    return model


def squeezenet_chess(image_shape=(8, 8, 18)):
    chessboard_before = tf.keras.Input(shape=image_shape, name='chessboard_before')
    chessboard_after = tf.keras.Input(shape=image_shape, name='chessboard_after')

    squeezenet_original1 = squeezenet(image_shape=image_shape)
    feature_extractor1 = Model(inputs=squeezenet_original1.inputs,
                               outputs=squeezenet_original1.get_layer('drop9').output,
                               name='feature_extractor1')

    squeezenet_original2 = squeezenet(image_shape=image_shape)
    feature_extractor2 = Model(inputs=squeezenet_original2.inputs,
                               outputs=squeezenet_original2.get_layer('drop9').output,
                               name='feature_extractor2')

    features_before = feature_extractor1(chessboard_before)
    features_after = feature_extractor2(chessboard_after)

    concatenated_features = Concatenate(axis=-1, name='concatenated_features')([features_before, features_after])
    x = Flatten()(concatenated_features)
    x = Dropout(0.25)(x)
    out = Dense(units=1, activation='sigmoid', name='out')(x)
    model = Model(inputs=[chessboard_before, chessboard_after], outputs=out)
    return model


if __name__ == '__main__':
    schess = squeezenet_chess(image_shape=(8, 8, 18))
    schess.summary()
