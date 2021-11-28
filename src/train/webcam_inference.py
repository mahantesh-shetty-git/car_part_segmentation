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

colors=\
{
    0:(0,0,255), # red
    1:(0,255,0), # green
    2:(255,0,0), #blue
    3:(0,0,128), #maroon
    4: (0,215,255), # gold
    5: (79,79,47), #slate gray
    6: (255,255,0), #aqua
    7: (128,0,0), #navy
    8: (226,43,138), # blue violet
    9: (255,0, 255), #fushia
    10: (19,69,139), #saddle brown
    11: (0,0,0), #black
    12: (128,0,128),
    13: (160,158,95),
    14: (128,128,0),
    15: (87,139,46),
    16: (92,92,205),
    17: (255,255,255),
    18: (147,20,255),


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

    # convert the image to rgb from bgr
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # normalize the value and set to type float
    frame = frame.astype('float')
    frame = frame/255.0

    frame = np.expand_dims(frame, axis=0)

    # call inference on image
    prediction = model.predict(frame)

    # convert it to the 2d mask again
    mask_image = np.argmax(prediction, axis=3)

    return mask_image

def infer_on_static_batch(model, config_dict):
    # include logic to predict on batch later

    frame = cv2.imread(r'C:\Users\mahan\PycharmProjects\car_part_segmentation\data\testset\JPEGImages\te96.jpg')

    frame_copy = frame.copy()
    h,w = frame.shape[:2]


    mask_image = inference_on_image(model, frame, config_dict)

    mask = mask_image[0]

    # segmentation class output
    segmentation = np.unique(mask)


    overlay_mask = np.zeros((config_dict['input_shape']+(3,)))
    b,g,r =cv2.split(overlay_mask)

    for seg_class in segmentation:
        coordinates = np.where(mask==seg_class)
        b[coordinates]=colors[seg_class][0]
        g[coordinates]=colors[seg_class][1]
        r[coordinates]=colors[seg_class][2]

    overlay_mask = cv2.merge((b,g,r))
    overlay_mask = overlay_mask.astype('uint8')
    overlay_mask = cv2.resize(overlay_mask, (w,h))

    added_image = cv2.addWeighted(frame_copy, 0.3, overlay_mask, 1, 0)

    filename = config_dict['load_model_path'].split('/')[-1]

    cv2.imwrite(f'{filename}.png', added_image)










if __name__=='__main__':
    config_dict = get_config_dict()

    # load the model for inference
    model = load_model(config_dict['load_model_path'], compile=False)

    #read the config file
    #webcam_inference(model=model, configuration=config_dict)

    infer_on_static_batch(model, config_dict)
