from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, \
    Activation, Conv2DTranspose, Concatenate
from tensorflow.keras import Model


class EfficientNetModel:
    def __init__(self, num_class=None, backbone='EfficientNetB0'):
        assert num_class is not None, "Number of class have not been specified"
        self.num_class = num_class


    def conv_block(self, inputs, num_filters):
        x = Conv2D(num_filters, 3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x

    def decoder_block(self, inputs, skip, num_filters):
        x = Conv2DTranspose(num_filters, (2,2), strides=2, padding='same')(inputs)
        x = Concatenate()([x, skip])
        x = self.conv_block(x, num_filters)
        return x

    def build_efficient_unet(self, input_shape):
        assert self.config_dict['input_shape'] == (input_shape[0],input_shape[1]), \
            "The size from input generator doesnt match model"

        # Define the input layer
        inputs = tf.keras.Input(input_shape)

        # Pre trained Encoder
        encoder = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)

        # get the skip connections
        s1 = encoder.get_layer("input_1").output
        s2 = encoder.get_layer("block2a_expand_activation").output
        s3 = encoder.get_layer("block3a_expand_activation").output
        s4 = encoder.get_layer("block4a_expand_activation").output

        # get the bottleneck layer
        b1 = encoder.get_layer("block6a_expand_activation").output

        # Decoder
        d1 = self.decoder_block(b1, s4, 512)
        d2 = self.decoder_block(d1, s3, 256)
        d3 = self.decoder_block(d2, s2, 128)
        d4 = self.decoder_block(d3, s1, 64)

        # Output
        outputs = Conv2D(self.num_class, 1, padding='same', activation='softmax')(d4)

        model = Model(inputs, outputs)

        return model


