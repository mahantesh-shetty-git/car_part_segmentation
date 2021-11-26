import cv2
import configparser

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
model_config = {
    'EfficientNetB0':
        {
            'model_name': 'EfficientNetB0',
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
            'model_name': 'ModelNetV2',
            'input_shape': (256, 256)
        }
}

def get_config_dict(mode='INFERENCE'):
    config = configparser.ConfigParser()
    config.read(r'../../config/train.ini')
    config_dict = dict(config.items(mode))

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


def webcam_inference(model, configuration=None):
    assert configuration is not None, "Pass configuration dictionary"
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()

        mask_image = inference_on_image(model, frame, config_dict)

        # show the mask
        plt.imshow(mask_image[0])
        plt.pause(0.01)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def inference_on_image(model, frame, config_dict):
    # preprocess to specific size
    frame = cv2.resize(frame, config_dict['input_shape'])

    frame = np.expand_dims(frame, axis=0)

    # call inference on image
    prediction = model.predict(frame)

    # convert it to the 2d mask again
    mask_image = np.argmax(prediction, axis=3)

    return mask_image

def infer_on_static_batch(model, config_dict):
    # include logic to predict on batch later

    frame = cv2.imread(r'C:\Users\mahan\PycharmProjects\car_part_segmentation\data\testset\JPEGImages\car10.jpg')

    mask_image = inference_on_image(model, frame, config_dict)

    print(np.unique(mask_image[0]))





if __name__=='__main__':
    config_dict = get_config_dict()

    # load the model for inference
    model = load_model(config_dict['load_model_path'])

    #read the config file
    #webcam_inference(model=model, configuration=config_dict)

    infer_on_static_batch(model, config_dict)
