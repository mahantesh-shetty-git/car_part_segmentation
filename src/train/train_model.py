# import all libraries
import configparser
import os
import argparse
import src.prepare_segmentation_mask.prepare_mask as data_generator
import src.model.efficientnet_model as EN_model

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
                    }
            }

    assert config_dict['backbone'] in model_config, "model information not available"
    config_dict['input_shape'] = model_config[config_dict['backbone']]['input_shape']
    return config_dict

class GetImageGenerators:
    def __init__(self, config_dict):
        self.train_annotations = config_dict['train_annotation_path']
        self.validation_annotations = config_dict['val_annotation_path']
        self.batch_size = config_dict['batch_size']

    def get_generator_object(self, annotation_path=None):
        assert annotation_path is not None, "No annotation path provided"
        data_gen = data_generator.DataManipulationCoco(annotation_path, self.batch_size)
        images = data_gen.get_all_combination_subset()
        generator_obj = data_gen.dataGeneratorCoco(images, None)
        return generator_obj

    def get_train_val_generator(self):
        train_gen = self.get_generator_object(self.train_annotations)
        val_gen = self.get_generator_object(self.validation_annotations)
        return train_gen, val_gen


if __name__=='__main__':    ## add logging and exception handling

    # add code to read the mode in which it is being run

    # read config into a dict
    config_dict = get_config_dict()

    # get the backbone to use for training
    backbone = config_dict['backbone']

    #  get the training and validation generator object
    gen_initializer = GetImageGenerators(config_dict)
    train_gen, val_gen = gen_initializer.get_train_val_generator()
    num_class =

    # get the model for training
    efficinet_class_init = EN_model.EfficientNetModel(num_class, backbone)
    efficient_model = efficinet_class_init.build_efficient_unet()       #need to passs input shape here

    # compile the model with loss function
    efficient_model.compile('adam', loss=config_dict['loss_func'], metrics=config_dict['metrics'])

    # Deal with class imbalance # have to incorporate it in
    if eval(config_dict['use_class_weight']):
        class_weights = None
    else:
        pass


    # fit the model
    efficient_model.fit(train_gen, batch_size=config_dict['batch_size'],
                        epochs=300, verbose=1,
                        validation_data=val_gen,
                        shuffle=False,
                        class_weight=class_weights
                        )


# # recheck the number of unique classses in your segmentation mask
# #np.unique(train_masks) where train_masks is np array
#
# from sklearn.preprocessing import LabelEncoder
# labelencoder = LabelEncoder()
# n, h, w = train_masks.shape
# train_masks_reshaped = train_masks.reshape(-1,1)
# train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
# train_masks_reshaped_encoded_orginal_shape = train_masks_reshaped_encoded.reshape(n,h,w)
#
# # make the dimension as 4
#
# # normalize the data by setting it to a float object with values between 0 and 1
#
# # get the train and validation data/ generators
#
# # convert to categorical
# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(train_masks, num_classes=N_CLASSES)
#
# # compute class-weights for class balancing
# from sklearn.utils import class_weight
# class_weight.compute_class_weight('balanced', unique_class_labels, train_masks)

#
