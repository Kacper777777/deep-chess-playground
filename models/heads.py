from tensorflow.keras.layers import Convolution2D, GlobalAveragePooling2D, Activation


def move_classification_head(features):
    x = Convolution2D(73, (1, 1), padding='valid', activation='relu')(features)
    out = Activation('softmax', name='move_probabilities')(x)
    return out


def position_evaluation_head(features):
    x = Convolution2D(3, (1, 1), padding='valid', activation='relu')(features)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='evaluation')(x)
    return out
