import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pycocotools.coco import  COCO
import numpy as np
import skimage.io as io
import cv2
import os
import random

train_json_path = r'../../data/trainingset/annotations.json'
val_json_path = r'../../data/testset/annotations.json'

def read_annotation_json():
    # make a file object for reading json path
    file_obj = open(train_json_path)

    # load the json data
    json_data = json.load(file_obj)

    # get all the keys
    keys = json_data.keys()

    usable_df_list = []

    for key in keys:
        temp_dict ={}
        try:
            df = pd.DataFrame(json_data[key])
            temp_dict['key']=key
            temp_dict['df'] = df

            print('-----------------------------------------')
            print(key)
            print(df.columns)
            print(df.head(5))


            usable_df_list.append(temp_dict)
        except:
            continue

    # close the file object
    file_obj.close()

    return usable_df_list

class DataManipulationCoco:
    def __init__(self, annotation_file=None, batch_size=None):
        self.coco = COCO(annotation_file)
        self.batch_size = batch_size
        self.num_class = self.get_number_of_class()

    def get_categories(self):
        category_ids = self.coco.getCatIds()
        self.categories = self.coco.loadCats(category_ids)
        return self.categories

    def get_number_of_class(self):
        classes = [category['name'] for category in self.get_categories()]
        return len(classes)


    def get_class_for_category_id(self, category_id):
        categories = self.get_categories()
        for i in range(categories):
            if categories[i]['id'] == category_id:
                return categories[i]['name']
        return None

    def get_all_combination_subset(self, filter_classes=None):
        images = []
        if filter_classes!=None:
            for f_class in filter_classes:
                category_id = self.coco.getCatIds(catNms=f_class)
                image_ids = self.coco.getImgIds(catIds=category_id)
                images += self.coco.loadImgs(image_ids)
        else:
            image_ids = self.coco.getImgIds()
            images = self.coco.loadImgs(image_ids)


        unique_images = []
        for i in range(len(images)):
            if images[i] not in unique_images:
                unique_images.append(images[i])
        print("Number of unique images ", len(unique_images))

        random.shuffle(unique_images)

        return unique_images

    def getClassName(self, classID, cats):
        for i in range(len(cats)):
            if cats[i]['id'] == classID:
                return cats[i]['name']
        return None

    def getImage(self, imageObj, img_folder, input_image_size):
        # Read and normalize an image
        train_img = io.imread(img_folder + '/' + imageObj['file_name']) / 255.0
        # Resize
        train_img = cv2.resize(train_img, input_image_size)
        if (len(train_img.shape) == 3 and train_img.shape[2] == 3):  # If it is a RGB 3 channel image
            return train_img
        else:  # To handle a black and white image, increase dimensions to 3
            stacked_img = np.stack((train_img,) * 3, axis=-1)
            return stacked_img

    def getNormalMask(self, imageObj, classes,  catIds, input_image_size):
        annIds = self.coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        cats = self.coco.loadCats(catIds)
        train_mask = np.zeros(input_image_size)
        classes = [category['name'] for category in self.get_categories()]
        for a in range(len(anns)):
            className = self.getClassName(anns[a]['category_id'], cats)
            pixel_value = classes.index(className) + 1
            new_mask = cv2.resize(self.coco.annToMask(anns[a]) * pixel_value, input_image_size)
            train_mask = np.maximum(new_mask, train_mask)

        # Add extra dimension for parity with train_img size [X * X * 3]
        train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
        return train_mask

    def getBinaryMask(self, imageObj, catIds, input_image_size):
        annIds = self.coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        train_mask = np.zeros(input_image_size)
        for a in range(len(anns)):
            new_mask = cv2.resize(self.coco.annToMask(anns[a]), input_image_size)

            # Threshold because resizing may cause extraneous values
            new_mask[new_mask >= 0.5] = 1
            new_mask[new_mask < 0.5] = 0

            train_mask = np.maximum(new_mask, train_mask)

        # Add extra dimension for parity with train_img size [X * X * 3]
        train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
        return train_mask

    def dataGeneratorCoco(self, images, classes=None,
                          input_image_size=(224, 224), batch_size=None,  mask_type='normal'):

        img_folder = r'../../data/trainingset/'
        dataset_size = len(images)
        if classes is None:
            classes = [category['name'] for category in self.get_categories()]
        catIds = self.coco.getCatIds(catNms=classes)
        print("This is catIds", catIds)
        batch_size=self.batch_size

        c = 0
        while (True):
            img = np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float')
            mask = np.zeros((batch_size, input_image_size[0], input_image_size[1], 1)).astype('float')

            for i in range(c, c + batch_size):  # initially from 0 to batch_size, when c = 0
                imageObj = images[i]

                ### Retrieve Image ###
                train_img = self.getImage(imageObj, img_folder, input_image_size)

                ### Create Mask ###
                if mask_type == "binary":
                    train_mask = self.getBinaryMask(imageObj, catIds, input_image_size)

                elif mask_type == "normal":
                    train_mask = self.getNormalMask(imageObj, classes, catIds, input_image_size)

                    # Add to respective batch sized arrays
                img[i - c] = train_img
                mask[i - c] = train_mask

            c += batch_size
            if (c + batch_size >= dataset_size):
                c = 0
                random.shuffle(images)
            yield img, mask

    def get_image_subset(self, filter_class=None):
        # Get all category class in case no class has been defined
        if filter_class==None:
            filter_class = [category['name'] for category in self.get_categories()]

        # get all the category ids
        category_ids = self.coco.getCatIds(catNms=filter_class)
        print(category_ids)

        self.get_all_combination_subset(filter_classes=filter_class)

        # get all images containing the above category ids
        image_ids = self.coco.getImgIds(catIds=category_ids)
        print(image_ids)

        # print number of images containing all classes
        print("Number of images containing all classes ", len(image_ids))

    def visualizeGenerator(self, gen):
        img, mask = next(gen)

        fig = plt.figure(figsize=(20, 10))
        outerGrid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)

        for i in range(2):
            innerGrid = gridspec.GridSpecFromSubplotSpec(2, 2,
                                                         subplot_spec=outerGrid[i], wspace=0.05, hspace=0.05)

            for j in range(4):
                ax = plt.Subplot(fig, innerGrid[j])
                if (i == 1):
                    ax.imshow(img[j])
                else:
                    ax.imshow(mask[j][:, :, 0])

                ax.axis('off')
                fig.add_subplot(ax)
        plt.savefig('graph.png')


if __name__ == '__main__':
    # usable_data_list = read_annotation_json()
    # for dict_obj in usable_data_list:
    #     if dict_obj['key']=='annotations':
    #         print(dict_obj['df']['category_id'].value_counts().plot.bar())
    #         print(dict_obj['df']['segmentation'])
    # plt.show()
    data_manipulation = DataManipulationCoco()
    images = data_manipulation.get_all_combination_subset()
    train_gen = data_manipulation.dataGeneratorCoco(images, None)
    data_manipulation.visualizeGenerator(train_gen)
