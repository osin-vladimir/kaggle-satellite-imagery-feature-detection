from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU
from keras.layers import BatchNormalization, Lambda, Dropout
from metrics import dice_coef, dice_coef_loss, jaccard_coef, jaccard_coef_loss, jaccard_coef_int

def rblock(inputs, num, depth, scale=0.1):
    residual = Convolution2D(depth, num, num, border_mode='same')(inputs)
    residual = BatchNormalization(mode=2, axis=1)(residual)
    residual = Lambda(lambda x: x*scale)(residual)
    res = _shortcut(inputs, residual)
    return ELU()(res)

def conv(data, n_filters, size):
    return ELU()(Convolution2D(n_filters, size, size, border_mode='same', init='he_normal')(data))

def _shortcut(_input, residual):
    stride_width = _input._keras_shape[2] / residual._keras_shape[2]
    stride_height = _input._keras_shape[3] / residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == _input._keras_shape[1]

    shortcut = _input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[1],
                                 nb_row=1,
                                 nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 init="he_normal",
                                 border_mode="valid")(_input)

    return merge([shortcut, residual], mode="sum")

def get_unet(ISZ, N_Cls):
    inputs = Input((8, ISZ, ISZ))

    conv1 = conv(inputs, 32, 3)
    conv1 = conv(conv1, 32, 3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.2)(pool1)

    conv2 = conv(pool1, 64, 3)
    conv2 = conv(conv2, 64, 3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.2)(pool2)

    conv3 = conv(pool2, 128, 3)
    conv3 = conv(conv3, 128, 3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.2)(pool3)

    conv4 = conv(pool3, 256, 3)
    conv4 = conv(conv4, 256, 3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.2)(pool4)

    conv5 = conv(pool4, 512, 3)
    conv5 = conv(conv5, 512, 3)
    conv5 = Dropout(0.2)(conv5)

    after_conv4 = rblock(conv4, 1, 256)
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), after_conv4], mode='concat', concat_axis=1)
    conv6 = conv(up6, 256, 3)
    conv6 = conv(conv6, 256, 3)
    conv6 = Dropout(0.2)(conv6)

    after_conv3 = rblock(conv3, 1, 128)
    up7 = merge([UpSampling2D(size=(2, 2))(conv6), after_conv3], mode='concat', concat_axis=1)
    conv7 = conv(up7, 128, 3)
    conv7 = conv(conv7, 128, 3)
    conv7 = Dropout(0.2)(conv7)

    after_conv2 = rblock(conv2, 1, 64)
    up8 = merge([UpSampling2D(size=(2, 2))(conv7), after_conv2], mode='concat', concat_axis=1)
    conv8 = conv(up8, 64, 3)
    conv8 = conv(conv8, 64, 3)
    conv8 = Dropout(0.2)(conv8)

    after_conv1 = rblock(conv1, 1, 32)
    up9 = merge([UpSampling2D(size=(2, 2))(conv8), after_conv1], mode='concat', concat_axis=1)
    conv9 = conv(up9, 32, 3)
    conv9 = conv(conv9, 32, 3)
    conv9 = Dropout(0.2)(conv9)

    conv10 = Convolution2D(N_Cls, 1, 1, init='he_normal', activation='sigmoid')(conv9)
    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=[jaccard_coef, dice_coef])
    return model