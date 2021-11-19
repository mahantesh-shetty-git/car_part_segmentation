# car_part_segmentation

The car part segmentation project uses the car part segmentation dataset published in the github repository: https://github.com/dsmlr/Car-Parts-Segmentation.git

It uses a pycoco dataset manipulation library installed from pypi for creating mask for semantic segmentation: https://github.com/philferriere/cocoapi.git

Unlike the data arranged in coco-dataset, the dataset is arranged as:

data
----trainingset
    |----JPEGImages
        |----*.jpg
    |----annotations.json
----testset
    |----JPEGImages
        |----*.jpg
    |----annotations.json
----doc_images
    ----*.webp

Finally, it has the model file to train efficientnet and mobilenet for realtime semantic segmentations.

