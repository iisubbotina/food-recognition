# %% [code] {"jupyter":{"outputs_hidden":false}}
import sys

print(sys.version)

import warnings
import tensorflow as tf
import os

my_token = 'Введите токен вашего тг бота'
warnings.filterwarnings('ignore')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:06:34.273874Z","iopub.execute_input":"2023-07-11T15:06:34.274624Z","iopub.status.idle":"2023-07-11T15:07:59.960461Z","shell.execute_reply.started":"2023-07-11T15:06:34.274577Z","shell.execute_reply":"2023-07-11T15:07:59.959179Z"}}


# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:10:25.181821Z","iopub.execute_input":"2023-07-11T15:10:25.182635Z","iopub.status.idle":"2023-07-11T15:10:25.187959Z","shell.execute_reply.started":"2023-07-11T15:10:25.182584Z","shell.execute_reply":"2023-07-11T15:10:25.187130Z"}}
import os

DATA_DIR = os.path.join("data", "train")

# Директория для сохранения логов и обученных моделей
ROOT_DIR = os.path.abspath(".")

# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:10:28.692235Z","iopub.execute_input":"2023-07-11T15:10:28.692882Z","iopub.status.idle":"2023-07-11T15:10:50.180594Z","shell.execute_reply.started":"2023-07-11T15:10:28.692827Z","shell.execute_reply":"2023-07-11T15:10:50.179533Z"}}

# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:10:56.255959Z","iopub.execute_input":"2023-07-11T15:10:56.256400Z","iopub.status.idle":"2023-07-11T15:11:18.037351Z","shell.execute_reply.started":"2023-07-11T15:10:56.256358Z","shell.execute_reply":"2023-07-11T15:11:18.036031Z"}}
 # subdirectory=PythonAPI

# %% [markdown]
# ### Импорт библиотек

# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:17:09.559338Z","iopub.execute_input":"2023-07-11T15:17:09.559793Z","iopub.status.idle":"2023-07-11T15:17:14.975565Z","shell.execute_reply.started":"2023-07-11T15:17:09.559753Z","shell.execute_reply":"2023-07-11T15:17:14.974432Z"}}
import sys

sys.path.append(os.path.join('.', 'Mask_RCNN'))  # To find local version of the library
sys.path.append(ROOT_DIR)
import sys
import re
import random
import pandas as pd
import os
import numpy as np
import mrcnn.model as modellib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import matplotlib
import math
import logging
import json
import itertools
import glob
import cv2
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from mrcnn.model import log
from mrcnn.config import Config
from mrcnn import visualize
from mrcnn import utils
from matplotlib.patches import Polygon
from imgaug import augmenters as iaa
from collections import defaultdict, Counter
from collections import OrderedDict

ROOT_DIR = os.path.abspath(".")


# %% [markdown]
# ### Определение класса набора данных и конфигурации

# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:17:44.434888Z","iopub.execute_input":"2023-07-11T15:17:44.435426Z","iopub.status.idle":"2023-07-11T15:17:44.474512Z","shell.execute_reply.started":"2023-07-11T15:17:44.435378Z","shell.execute_reply":"2023-07-11T15:17:44.473131Z"}}
class FoodChallengeDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, load_small=False, return_coco=True):
        """ Loads dataset released for the AICrowd Food Challenge
            Params:
                - dataset_dir : root directory of the dataset (can point to the train/val folder)
                - load_small : Boolean value which signals if the annotations for all the images need to be loaded into the memory,
                               or if only a small subset of the same should be loaded into memory
        """
        self.load_small = load_small
        if self.load_small:
            annotation_path = os.path.join(dataset_dir, "annotation-small.json")
        else:
            annotation_path = os.path.join(dataset_dir, "annotations.json")

        image_dir = os.path.join(dataset_dir, "images")
        print("Annotation Path ", annotation_path)
        print("Image Dir ", image_dir)
        assert os.path.exists(annotation_path) and os.path.exists(image_dir)

        self.coco = COCO(annotation_path)
        self.image_dir = image_dir

        # Load all classes (Only Building in this version)
        classIds = self.coco.getCatIds()

        # Load all images
        image_ids = list(self.coco.imgs.keys())

        # register classes
        for _class_id in classIds:
            self.add_class("crowdai-food-challenge", _class_id, self.coco.loadCats(_class_id)[0]["name"])

        # Register Images
        for _img_id in image_ids:
            assert (os.path.exists(os.path.join(image_dir, self.coco.imgs[_img_id]['file_name'])))
            self.add_image(
                "crowdai-food-challenge", image_id=_img_id,
                path=os.path.join(image_dir, self.coco.imgs[_img_id]['file_name']),
                width=self.coco.imgs[_img_id]["width"],
                height=self.coco.imgs[_img_id]["height"],
                annotations=self.coco.loadAnns(self.coco.getAnnIds(
                    imgIds=[_img_id],
                    catIds=classIds,
                    iscrowd=None)))

        if return_coco:
            return self.coco

    def load_mask(self, image_id):
        """ Loads instance mask for a given image
              This function converts mask from the coco format to a
              a bitmap [height, width, instance]
            Params:
                - image_id : reference id for a given image

            Returns:
                masks : A bool array of shape [height, width, instances] with
                    one mask per instance
                class_ids : a 1D array of classIds of the corresponding instance masks
                    (In this version of the challenge it will be of shape [instances] and always be filled with the class-id of the "Building" class.)
        """

        image_info = self.image_info[image_id]
        assert image_info["source"] == "crowdai-food-challenge"

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "crowdai-food-challenge.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue

                # Ignore the notion of "is_crowd" as specified in the coco format
                # as we donot have the said annotation in the current version of the dataset

                instance_masks.append(m)
                class_ids.append(class_id)
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(FoodChallengeDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a reference for a particular image

            Ideally you this function is supposed to return a URL
            but in this case, we will simply return the image_id
        """
        return "crowdai-food-challenge::{}".format(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:17:52.565915Z","iopub.execute_input":"2023-07-11T15:17:52.566373Z","iopub.status.idle":"2023-07-11T15:17:52.579184Z","shell.execute_reply.started":"2023-07-11T15:17:52.566335Z","shell.execute_reply":"2023-07-11T15:17:52.578141Z"}}
class FoodChallengeConfig(Config):
    """Configuration for training on data in MS COCO format.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "crowdai-food-challenge"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 5

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 41  # 1 Backgroun + 1 Building

    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50

    IMAGE_MAX_DIM = 256
    IMAGE_MIN_DIM = 256


config = FoodChallengeConfig()
config.display()

# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:18:21.422984Z","iopub.execute_input":"2023-07-11T15:18:21.423452Z","iopub.status.idle":"2023-07-11T15:19:24.169653Z","shell.execute_reply.started":"2023-07-11T15:18:21.423409Z","shell.execute_reply":"2023-07-11T15:19:24.168345Z"}}


# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:19:28.821902Z","iopub.execute_input":"2023-07-11T15:19:28.822682Z","iopub.status.idle":"2023-07-11T15:19:29.944815Z","shell.execute_reply.started":"2023-07-11T15:19:28.822633Z","shell.execute_reply":"2023-07-11T15:19:29.943374Z"}}
import shutil
import os

src = os.path.join('data', 'train', 'train', 'annotations.json')
dst = os.path.join('data', 'train', 'train', 'annotation.json')
# Новый путь и имя файла

if os.path.exists(src):
    shutil.copy(src, dst)
    print(f"Файл '{os.path.basename(src)}' успешно скопирован в '{dst}'.")
else:
    print(f"Файл '{src}' не найден.")

# %% [markdown]
# ### Загрузка датасета

# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:19:37.812615Z","iopub.execute_input":"2023-07-11T15:19:37.813399Z","iopub.status.idle":"2023-07-11T15:19:38.753199Z","shell.execute_reply.started":"2023-07-11T15:19:37.813352Z","shell.execute_reply":"2023-07-11T15:19:38.751114Z"}}
# from mrcnn.dataset import FoodChallengeDataset
dataset_train = FoodChallengeDataset()
dataset_train.load_dataset(dataset_dir=os.path.join("data", "train", "train"),
                           load_small=False)
# dataset_train.load_dataset(dataset_dir="train", load_small=False)
dataset_train.prepare()
dataset = dataset_train


# %% [markdown]
# ## Моделирование с помощью Mask-RCNN

# %% [code] {"execution":{"iopub.status.busy":"2023-07-09T11:57:09.326680Z","iopub.execute_input":"2023-07-09T11:57:09.327116Z","iopub.status.idle":"2023-07-09T11:57:10.346806Z","shell.execute_reply.started":"2023-07-09T11:57:09.327076Z","shell.execute_reply":"2023-07-09T11:57:10.345150Z"}}
import h5py

# %% [code] {"execution":{"iopub.status.busy":"2023-07-09T11:57:17.071324Z","iopub.execute_input":"2023-07-09T11:57:17.071963Z","iopub.status.idle":"2023-07-09T11:57:22.553385Z","shell.execute_reply.started":"2023-07-09T11:57:17.071900Z","shell.execute_reply":"2023-07-09T11:57:22.552060Z"}}


# %% [code] {"execution":{"iopub.status.busy":"2023-07-09T11:57:27.151149Z","iopub.execute_input":"2023-07-09T11:57:27.151543Z","iopub.status.idle":"2023-07-09T11:57:41.122913Z","shell.execute_reply.started":"2023-07-09T11:57:27.151506Z","shell.execute_reply":"2023-07-09T11:57:41.121841Z"}}
from tensorflow.keras import backend as K

backend = K.backend()



# Патч для h5py, чтобы избежать ошибки decode()
# Открытие файла HDF5 с разрешением на чтение и запись


# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:21:55.490955Z","iopub.execute_input":"2023-07-11T15:21:55.491417Z","iopub.status.idle":"2023-07-11T15:21:56.728876Z","shell.execute_reply.started":"2023-07-11T15:21:55.491378Z","shell.execute_reply":"2023-07-11T15:21:56.727824Z"}}

dataset_val = FoodChallengeDataset()
val_coco = dataset_val.load_dataset(dataset_dir=os.path.join("data", "val", "val"),
                                    load_small=False, return_coco=True)
dataset_val.prepare()

# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:20:50.435954Z","iopub.execute_input":"2023-07-11T15:20:50.436421Z","iopub.status.idle":"2023-07-11T15:20:50.441587Z","shell.execute_reply.started":"2023-07-11T15:20:50.436380Z","shell.execute_reply":"2023-07-11T15:20:50.440340Z"}}

model_path = os.path.join("data", "final_model.h5")



# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:20:56.099286Z","iopub.execute_input":"2023-07-11T15:20:56.099742Z","iopub.status.idle":"2023-07-11T15:20:56.112580Z","shell.execute_reply.started":"2023-07-11T15:20:56.099700Z","shell.execute_reply":"2023-07-11T15:20:56.111334Z"}}
class InferenceConfig(FoodChallengeConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 41  # 1 Background + 61 classes
    IMAGE_MAX_DIM = 320
    IMAGE_MIN_DIM = 320
    NAME = "crowdai-food-challenge"
    DETECTION_MIN_CONFIDENCE = 0.2


inference_config = InferenceConfig()
inference_config.display()

# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:21:01.782504Z","iopub.execute_input":"2023-07-11T15:21:01.782947Z","iopub.status.idle":"2023-07-11T15:21:18.080391Z","shell.execute_reply.started":"2023-07-11T15:21:01.782907Z","shell.execute_reply":"2023-07-11T15:21:18.079141Z"}}
# Recreate the model in inference mode
model = modellib.MaskRCNN(mode='inference',
                          config=inference_config,
                          model_dir=ROOT_DIR)

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
#print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-09T11:58:28.134778Z","iopub.execute_input":"2023-07-09T11:58:28.135515Z","iopub.status.idle":"2023-07-09T11:58:38.843407Z","shell.execute_reply.started":"2023-07-09T11:58:28.135468Z","shell.execute_reply":"2023-07-09T11:58:38.842067Z"}}
# Show few example of ground truth vs. predictions on the validation dataset
dataset = dataset_val
#fig = plt.figure(figsize=(10, 30))

for i in range(4):
    image_id = random.choice(dataset.image_ids)

    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)

    #print(original_image.shape)
    #plt.subplot(6, 2, 2 * i + 1)
    #visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                #dataset.class_names, ax=fig.axes[-1])

    #plt.subplot(6, 2, 2 * i + 2)
    results = model.detect([original_image])  # , verbose=1)
    r = results[0]
    #isualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                #dataset.class_names, r['scores'], ax=fig.axes[-1])


# %% [markdown]
#  ## Тестирование модели на изображении


# %% [markdown]
# ### Функция для детекции продуктов на одном фото

# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:22:22.147432Z","iopub.execute_input":"2023-07-11T15:22:22.148163Z","iopub.status.idle":"2023-07-11T15:22:22.163607Z","shell.execute_reply.started":"2023-07-11T15:22:22.148119Z","shell.execute_reply":"2023-07-11T15:22:22.162070Z"}}

def predict_image(image):
    # Загрузка изображения с помощью PIL
    # image = Image.open(image_path)

    # Изменение размера изображения до 320x320
    resized_image = image.resize((320, 320))

    # Используйте измененное изображение для предсказания модели
    results = model.detect([np.array(resized_image)])
    r = results[0]

    # Создание графика с размером исходного изображения
    fig = plt.figure(figsize=(image.width / 80, image.height / 80))
    ax1 = fig.add_subplot(1, 1, 1)

    # Отображение исходного изображения
    ax1.imshow(resized_image)
    ax1.axis('off')

    # Отображение результатов предсказания
    visualize.display_instances(np.array(resized_image), r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'], ax=ax1)

    # Удаление осей из графика
    ax1.axis('off')

    # Сохранение графика в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    # Преобразование буфера в изображение PIL
    result_image = Image.open(buf)

    # Изменение размера результата обратно на исходный размер
    result_image = result_image.resize(image.size)

    # Закрытие графика
    plt.close()

    # Возврат фотографии и текста названий классов
    class_names = [dataset_val.class_names[class_id] for class_id in r['class_ids']]
    return result_image, class_names

# %% [markdown]
# ## Реализация телеграмм бота

# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:22:26.493436Z","iopub.execute_input":"2023-07-11T15:22:26.493885Z","iopub.status.idle":"2023-07-11T15:23:01.874034Z","shell.execute_reply.started":"2023-07-11T15:22:26.493837Z","shell.execute_reply":"2023-07-11T15:23:01.872466Z"}}


# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:23:21.688953Z","iopub.execute_input":"2023-07-11T15:23:21.689637Z","iopub.status.idle":"2023-07-11T15:23:21.860155Z","shell.execute_reply.started":"2023-07-11T15:23:21.689588Z","shell.execute_reply":"2023-07-11T15:23:21.859123Z"}}
from deep_translator import GoogleTranslator
import telebot
from telebot import types
from PIL import Image
import matplotlib.pyplot as plt
import io

# %% [markdown]
# Создание объекта класса для перевода названий продуктов с английского на русский:

# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:23:24.965164Z","iopub.execute_input":"2023-07-11T15:23:24.965576Z","iopub.status.idle":"2023-07-11T15:23:24.984867Z","shell.execute_reply.started":"2023-07-11T15:23:24.965541Z","shell.execute_reply":"2023-07-11T15:23:24.983417Z"}}
translator = GoogleTranslator(source='en', target='ru')

# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:23:27.729420Z","iopub.execute_input":"2023-07-11T15:23:27.729851Z","iopub.status.idle":"2023-07-11T15:23:27.736962Z","shell.execute_reply.started":"2023-07-11T15:23:27.729815Z","shell.execute_reply":"2023-07-11T15:23:27.735304Z"}}
bot = telebot.TeleBot(my_token)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:23:30.740395Z","iopub.execute_input":"2023-07-11T15:23:30.740975Z","iopub.status.idle":"2023-07-11T15:23:30.746564Z","shell.execute_reply.started":"2023-07-11T15:23:30.740937Z","shell.execute_reply":"2023-07-11T15:23:30.745590Z"}}
@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, "Для распознавания еды отправьте фото")


# %% [markdown]
# ### Функция для подсчета калорий

# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:23:35.984078Z","iopub.execute_input":"2023-07-11T15:23:35.984662Z","iopub.status.idle":"2023-07-11T15:23:36.038910Z","shell.execute_reply.started":"2023-07-11T15:23:35.984601Z","shell.execute_reply":"2023-07-11T15:23:36.037790Z"}}
import pandas as pd


def find_food_calories(food_list):
    # Загрузка данных из CSV-файла
    data = pd.read_csv(os.path.join("data", "Food and Calories - Sheet1.csv"))

    total_calories = 0
    results = []
    for food in food_list:
        filtered_data = data[data['Food'].str.contains(food, case=False, na=False)]
        print(f"Поиск для {food}: найдено {len(filtered_data)} совпадений")
        if not filtered_data.empty:
            calories_str = filtered_data['Calories'].iloc[0]
            calories = int(calories_str.split(' ')[0])  # Удаление "cal" и преобразование в int
            total_calories += calories
            result_text = f"Food: {filtered_data['Food'].iloc[0]}\nServing: {filtered_data['Serving'].iloc[0]}\nCalories: {calories}"
            results.append(result_text)

    result_text = '\n\n'.join(results)
    result_text += f"\n\nTotal Calories: {total_calories}"
    return result_text


# %% [markdown]
# ### Функция для поиска рецептов:

# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:24:10.951574Z","iopub.execute_input":"2023-07-11T15:24:10.952030Z","iopub.status.idle":"2023-07-11T15:24:13.112086Z","shell.execute_reply.started":"2023-07-11T15:24:10.951978Z","shell.execute_reply":"2023-07-11T15:24:13.110987Z"}}
import pandas as pd


def search_recipes(ingredients_list):
    # Загрузка данных из CSV-файла
    data = pd.read_csv(os.path.join("data", "povarenok_recipes_2021_06_16.csv"))

    # Замена значений типа float на пустую строку
    data['ingredients'] = data['ingredients'].fillna('')

    # Приведение ингредиентов к нижнему регистру
    ingredients_list_lower = [ingredient.lower() for ingredient in ingredients_list]

    # Фильтрация данных по столбцу "ingredients"
    filtered_data = data[data['ingredients'].apply(
        lambda x: all(ingredient.lower() in x.lower() for ingredient in ingredients_list_lower))]

    # Ограничение списка до 15 рецептов
    filtered_data = filtered_data.head(5)

    # Возвращение найденных рецептов в виде текста
    recipes_text = ""
    for index, row in filtered_data.iterrows():
        recipes_text += f"Рецепт #{index + 1}:\n"
        recipes_text += f"URL: {row['url']}\n"
        recipes_text += f"Название: {row['name']}\n"
        recipes_text += f"Ингредиенты: {row['ingredients']}\n\n"

    return recipes_text


# %% [markdown]
# ### Реализация функций телеграмм бота
#
# Реализованы функции:
# * Получение фотографии от пользователя и отправление ему предсказания вместе с измененным фото
# * Изменение списка ингредиентов: добавление и удаление продуктов
# * Функция подсчета калорий: данные из датасета food-and-their-calories
# * Функция получения рецептов по ингредиентам: данные из датасета povarenok-recipes

# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:24:01.189181Z","iopub.execute_input":"2023-07-11T15:24:01.189949Z","iopub.status.idle":"2023-07-11T15:24:01.236639Z","shell.execute_reply.started":"2023-07-11T15:24:01.189897Z","shell.execute_reply":"2023-07-11T15:24:01.234773Z"}}
class_names_russian = []
user_step = {}


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    global class_names_russian
    class_names_russian = []
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    with open("image.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)

    with open("image.jpg", 'rb') as photo_file:
        image = Image.open(photo_file)
        result_image, class_names = predict_image(image)

        # Сохранение результатов предсказания в файл
        result_image.save("result_image.png")

        # Отправка результата пользователю
        with open("result_image.png", 'rb') as result_file:
            bot.send_photo(message.chat.id, result_file)

        # Перевод названий классов на русский язык
        for i, class_name in enumerate(class_names):
            if class_name == "chips-french-fries":
                class_names_russian.append("картофель-фри")
            elif class_name == "leaf-spinach":
                class_names_russian.append("листья шпината")
            elif class_name == "salad-leaf-salad-green":
                class_names_russian.append("салат зеленый")
            else:
                translated_text = translator.translate(class_name, lang_tgt='ru')
                class_names_russian.append(translated_text)

        # Формирование текста с переведенными названиями классов
        class_names_text = "Продукты, которые удалось обнаружить:\n"
        for i, class_name in enumerate(class_names_russian):
            class_names_text += f"{i + 1}. {class_name}\n"

        # Отправка текста названий классов пользователю
        bot.send_message(message.chat.id, class_names_text)

        bot.send_message(message.chat.id, "Хотите ли вы изменить список?", reply_markup=create_choice_keyboard())


def create_choice_keyboard():
    keyboard = types.ReplyKeyboardMarkup(row_width=2, one_time_keyboard=True)
    keyboard.add(types.KeyboardButton("Добавить продукт"))
    keyboard.add(types.KeyboardButton("Убрать продукт"))
    keyboard.add(types.KeyboardButton("Готово"))
    return keyboard


@bot.message_handler(func=lambda message: message.text == "Добавить продукт")
def handle_add_product(message):
    bot.send_message(message.chat.id, "Введите название продукта:")
    user_step[message.chat.id] = "add"


@bot.message_handler(func=lambda message: message.text == "Убрать продукт")
def handle_remove_product(message):
    bot.send_message(message.chat.id, "Введите номер продукта:")
    user_step[message.chat.id] = "remove"


@bot.message_handler(func=lambda message: message.text == "Найти рецепты с этими ингредиентами")
def handle_search_recipes(message):
    global class_names_russian

    # Вызов функции search_recipes() с передачей списка продуктов
    results = search_recipes(class_names_russian)

    # Отправка результатов пользователю
    print(class_names_russian)
    print(results)
    bot.send_message(message.chat.id, f"Рецепты с этими ингредиентами:\n{results}")


@bot.message_handler(func=lambda message: message.text == "Посчитать калории")
def handle_calculate_calories(message):
    global class_names_russian

    # Перевод элементов списка class_names_russian на английский
    translatorE = GoogleTranslator(source='ru', target='en')
    class_names_english = [translatorE.translate(name) for name in class_names_russian]

    # Вызов функции find_food_calories() с передачей списка продуктов
    result2 = find_food_calories(class_names_english)

    # Отправка результата пользователю
    bot.send_message(message.chat.id, f"Результат подсчета калорий:\n{result2}")


@bot.message_handler(func=lambda message: message.text == "Готово")
def handle_finished(message):
    bot.send_message(message.chat.id, "Список продуктов сохранен.")
    keyboard = types.ReplyKeyboardMarkup(row_width=2, one_time_keyboard=True)
    keyboard.add(types.KeyboardButton("Найти рецепты с этими ингредиентами"))
    keyboard.add(types.KeyboardButton("Посчитать калории"))

    bot.send_message(message.chat.id, "Что вы хотите сделать дальше?", reply_markup=keyboard)

    user_step[message.chat.id] = None


@bot.message_handler(func=lambda message: user_step.get(message.chat.id) == "add")
def add_product(message):
    global class_names_russian
    product_name = message.text
    class_names_russian.append(product_name)

    updated_list_text = "Измененный список продуктов:\n"
    for i, class_name in enumerate(class_names_russian):
        updated_list_text += f"{i + 1}. {class_name}\n"

    bot.send_message(message.chat.id, updated_list_text)
    bot.send_message(message.chat.id, "Хотите ли вы продолжить изменение списка?",
                     reply_markup=create_choice_keyboard())


@bot.message_handler(func=lambda message: user_step.get(message.chat.id) == "remove")
def remove_product(message):
    global class_names_russian
    try:
        product_number = int(message.text)

        if product_number > 0 and product_number <= len(class_names_russian):
            class_names_russian.pop(product_number - 1)

            updated_list_text = "Измененный список продуктов:\n"
            for i, class_name in enumerate(class_names_russian):
                updated_list_text += f"{i + 1}. {class_name}\n"

            bot.send_message(message.chat.id, updated_list_text)
            bot.send_message(message.chat.id, "Хотите ли вы продолжить изменение списка?",
                             reply_markup=create_choice_keyboard())

        else:
            bot.send_message(message.chat.id, "Неверный номер продукта. Пожалуйста, введите существующий номер.")

    except ValueError:
        bot.send_message(message.chat.id, "Неверный формат номера продукта. Пожалуйста, введите число.")


# %% [markdown]
# ### Запуск бота

# %% [code] {"execution":{"iopub.status.busy":"2023-07-11T15:30:52.029333Z","iopub.execute_input":"2023-07-11T15:30:52.029732Z"}}
print('Бот запущен')
bot.polling(none_stop=True, interval=0)
