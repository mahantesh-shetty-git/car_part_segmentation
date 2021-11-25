from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, UpSampling2D\
    ,Concatenate, Conv2D, BatchNormalization, Activation
from tensorflow.keras import Model

class MobilenetModels:
    def __init__(self, num_class=None):
        assert num_class is not None, "segmentation class not provided"
        self.num_class = num_class

    def build_mobilenet_model(self, input_shape):
        #
        inputs = Input(input_shape, name="input_image")

        # pretrained encoder
        encoder = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False, alpha=0.35)
        skip_connection_names = ['input_image', 'block_1_expand_relu','block_3_expand_relu',
                                 'block_6_expand_relu']
        encoder_output = encoder.get_layer('block_13_expand_relu').output  #(16, 16)

        f = [16, 32, 48, 64]
        x = encoder_output

        for i in range(1, len(f)+1, 1):
            x_skip = encoder.get_layer(skip_connection_names[-i]).output
            x = UpSampling2D((2, 2))(x)
            x = Concatenate()([x,x_skip])

            x = Conv2D(f[-i], (3,3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu') (x)

            x = Conv2D(f[-i], (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        outputs = Conv2D(self.num_class, 1, padding='same', activation='softmax')(x)

        model = Model(inputs, outputs)

        return model



