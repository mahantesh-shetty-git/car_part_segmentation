# import all libraries
import configparser
import datetime
import os
import sys
import segmentation_models as sm
import tensorflow.keras.losses

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse


import src.prepare_segmentation_mask.prepare_mask as data_generator
import src.model.efficientnet_model as EN_model
import src.model.mobilenet_model as MN_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

def get_config_dict(train_mode='TRAIN'):
    print("in configuration ", os.getcwd())
    config = configparser.ConfigParser()
    config.read(r'../../config/train.ini')
    config_dict = dict(config.items(train_mode))

    model_config = {
                'EfficientNetB0':
                    {
                    'model_name':'EfficientNetB0',
                    'input_shape': (224, 224)
                    },
                'EfficientNetB1':
                    {
                        'model_name': 'EfficientNetB1',
                        'input_shape': (240, 240)
                    },
                'EfficientNetB2':
                    {
                        'model_name': 'EfficientNetB2',
                        'input_shape': (260, 260)
                    },
                'EfficientNetB3':
                    {
                        'model_name': 'EfficientNetB3',
                        'input_shape': (300, 300)
                    },
                'EfficientNetB4':
                    {
                        'model_name': 'EfficientNetB4',
                        'input_shape': (380, 380)
                    },
                'EfficientNetB5':
                    {
                        'model_name': 'EfficientNetB5',
                        'input_shape': (456, 456)
                    },
                'EfficientNetB6':
                    {
                        'model_name': 'EfficientNetB6',
                        'input_shape': (528, 528)
                    },
                'EfficientNetB7':
                    {
                        'model_name': 'EfficientNetB7',
                        'input_shape': (600, 600)
                    },
                'MobileNetV2':
                    {
                        'model_name':'ModelNetV2',
                        'input_shape': (256, 256)
                    }
            }
    assert config_dict['backbone'] in model_config, "model information not available"
    config_dict['input_shape'] = model_config[config_dict['backbone']]['input_shape']
    return config_dict

class GetImageGenerators:

    def __init__(self, config_dict):
        self.train_annotations = config_dict['train_annotation_path']
        self.validation_annotations = config_dict['val_annotation_path']
        self.batch_size = int(config_dict['batch_size'])
        self.set_flag = False
        self.input_shape = config_dict['input_shape']

    def get_generator_object(self, annotation_path=None, training=False):
        assert annotation_path is not None, "No annotation path provided"
        data_gen = data_generator.DataManipulationCoco(annotation_path, self.batch_size, training=training)
        if not self.set_flag:
            self.num_class = data_gen.get_number_of_class()
            self.set_flag = True
        images = data_gen.get_all_combination_subset()
        generator_obj = data_gen.dataGeneratorCoco(images, input_image_size=self.input_shape)
        return generator_obj

    def get_train_val_generator(self):
        train_gen = self.get_generator_object(self.train_annotations, training=True)
        val_gen = self.get_generator_object(self.validation_annotations)
        return train_gen, val_gen

    def get_num_of_class(self):
        return self.num_class


if __name__=='__main__':    ## add logging and exception handling

    # add code to read the mode in which it is being run

    # read config into a dict
    config_dict = get_config_dict()

    # get the backbone to use for training
    backbone = config_dict['backbone']

    #  get the training and validation generator object
    gen_initializer = GetImageGenerators(config_dict)
    train_gen, val_gen = gen_initializer.get_train_val_generator()
    num_class = gen_initializer.get_num_of_class()

    img, mask = next(train_gen)
    val_img, val_mask = next(val_gen)
    print(f"input image shape is {img.shape} and input mask shape {mask.shape}")
    print(f"validation input image_shape {val_img.shape} and val mask shape is {val_mask.shape}")
    print(f"number of class is {num_class}")
    # sys.exit()

    input_size = config_dict['input_shape']+(3,)

    ## put this under decorator

    # get the model for training
    # efficinet_class_init = EN_model.EfficientNetModel(num_class, backbone)
    # model = efficinet_class_init.build_efficient_unet(input_size)       #need to passs input shape here
    # #
    mobilenet_class_init = MN_model.MobilenetModels(num_class)
    model = mobilenet_class_init.build_mobilenet_model(input_size)

    focal_loss = sm.losses.CategoricalFocalLoss(gamma=10)

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    model.compile('adam', loss=focal_loss, metrics='accuracy')

    # Deal with class imbalance # have to incorporate it in
    if eval(config_dict['use_class_weight']):
        class_weights = None
    else:
        pass

    # set callbacks
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_ckpt_dir = r'../../models/' + config_dict['backbone']+ time
    tboard_log_dir = r'../../log_dir/tboard_log_dir/' + config_dict['backbone']+time
    callbacks = [TensorBoard(tboard_log_dir, histogram_freq=1),
                 ModelCheckpoint(model_ckpt_dir, monitor='val_loss'
                                 , save_best_only=True,
                                 ),
                 ]

    # fit the model
    model.fit(train_gen, batch_size=config_dict['batch_size'],
                        epochs=300, validation_data=val_gen,
                        shuffle=False, callbacks=callbacks,
                        steps_per_epoch=100, validation_steps=25,
                        validation_batch_size=config_dict['batch_size'],

                        )


#
