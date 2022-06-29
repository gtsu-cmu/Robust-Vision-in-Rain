##################################################
# Local libraries

from detectron2_predictor import Detectron2Predictor
from detectron2_trainer import Detectron2Trainer
from detectron2_dataset import Detectron2CustomDataset
from utilities import create_file_list, imshow_jupyter
from datasets import get_carla_file_list, get_carla_dicts, get_cityscapes_file_list, get_cityscapes_dicts
from datasets import convert_carla, convert_cityscapes

##################################################
# Libraries

import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt
from PIL import Image

import os, json, cv2, random

##################################################
# Detectron2

import detectron2

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'

data_dir = '/home/gregory/Documents/RainPerception/'
EVAL_LEVEL_LIST = ['H', 'M', 'S']

data_carla_dir = data_dir + 'Carla/packaging/'

# Dir structure:
# <data_carla_dir>
#     packages2
#     packages3
#     ...

data_carla_train_file_list = get_carla_file_list(data_carla_dir, packages=['package2', 'package3', 'package4', 'package5', 'package6', 'package7', 'package9'], levels=['H', 'M', 'S'])
#data_carla_train_file_list = get_carla_file_list(data_carla_dir, packages=['package2'], levels=['H', 'M', 'S'])
# data_carla_val_file_list = get_carla_file_list(data_carla_dir, packages=['package8'], levels=['H', 'M', 'S'])
data_carla_val_file_list = get_carla_file_list(data_carla_dir, packages=['package8'], levels=EVAL_LEVEL_LIST)

print('**************************************************')
print('Carla clear')

def get_carla_clear_train_dicts():
    return get_carla_dicts(data_carla_train_file_list, data_carla_dir, clear=True, rain=False)

def get_carla_clear_val_dicts():
    return get_carla_dicts(data_carla_val_file_list, data_carla_dir, clear=True, rain=False)

print(len(get_carla_clear_train_dicts())) # 2990
print(len(get_carla_clear_val_dicts()))   # 1173

print('\n**************************************************')
print('Carla rain')

def get_carla_rain_train_dicts():
    return get_carla_dicts(data_carla_train_file_list, data_carla_dir, clear=False, rain=True)

def get_carla_rain_val_dicts():
    return get_carla_dicts(data_carla_val_file_list, data_carla_dir, clear=False, rain=True)

print(len(get_carla_rain_train_dicts())) # 2990
print(len(get_carla_rain_val_dicts()))   # 1173

print('\n**************************************************')
print('Carla all')

def get_carla_all_train_dicts():
    return get_carla_dicts(data_carla_train_file_list, data_carla_dir, clear=True, rain=True)

def get_carla_all_val_dicts():
    return get_carla_dicts(data_carla_val_file_list, data_carla_dir, clear=True, rain=True)

print(len(get_carla_all_train_dicts())) # 2990*2 = 5980
print(len(get_carla_all_val_dicts()))   # 1173*2 = 2346

data_carla_night_dir = data_dir + 'CarlaNight/night_packaging/'

# Dir structure:
# <data_carla_night_dir>
#     packages10
#     packages11
#     ...

data_carla_night_train_file_list = get_carla_file_list(data_carla_night_dir, packages=['package10', 'carla_data_night_1'], levels=['H', 'M', 'S'])
# data_carla_night_val_file_list = get_carla_file_list(data_carla_night_dir, packages=['package11'], levels=['H', 'M', 'S'])
data_carla_night_val_file_list = get_carla_file_list(data_carla_night_dir, packages=['package11', 'carla_data_night_2'], levels=EVAL_LEVEL_LIST)

#

print('**************************************************')
print('Carla night clear')

def get_carla_night_clear_train_dicts():
    return get_carla_dicts(data_carla_night_train_file_list, data_carla_night_dir, clear=True, rain=False)

def get_carla_night_clear_val_dicts():
    return get_carla_dicts(data_carla_night_val_file_list, data_carla_night_dir, clear=True, rain=False)

print(len(get_carla_night_clear_train_dicts())) # 2842
print(len(get_carla_night_clear_val_dicts()))   # 2974

print('\n**************************************************')
print('Carla night rain')

def get_carla_night_rain_train_dicts():
    return get_carla_dicts(data_carla_night_train_file_list, data_carla_night_dir, clear=False, rain=True)

def get_carla_night_rain_val_dicts():
    return get_carla_dicts(data_carla_night_val_file_list, data_carla_night_dir, clear=False, rain=True)
 
print(len(get_carla_night_rain_train_dicts())) # 2842
print(len(get_carla_night_rain_val_dicts()))   # 2974

print('\n**************************************************')
print('Carla night all')

def get_carla_night_all_train_dicts():
    return get_carla_dicts(data_carla_night_train_file_list, data_carla_night_dir, clear=True, rain=True)

def get_carla_night_all_val_dicts():
    return get_carla_dicts(data_carla_night_val_file_list, data_carla_night_dir, clear=True, rain=True)

print(len(get_carla_night_all_train_dicts())) # 2842*2 = 5684
print(len(get_carla_night_all_val_dicts()))   # 2974*2 = 5948

data_dir_cityscapes = data_dir + 'Cityscapes/leftImg8bit/train/'
# data_dir_rain_addition = data_dir + 'RainAddition/train/'
data_dir_rain_addition = data_dir + 'RainRemoval/RainAddition/'
anno_dir_cityscapes = data_dir + 'Cityscapes/gtFine/train/'
# anno_dir_cityscapes = data_dir + 'Cityscapes/mapped_labels/train/'

city_name_list, _ = create_file_list(data_dir_cityscapes)
print(f'Number of cities: {len(city_name_list)}')

data_cityscapes_train_file_list = get_cityscapes_file_list(data_dir_cityscapes, cities=city_name_list[:13])
data_cityscapes_val_file_list   = get_cityscapes_file_list(data_dir_cityscapes, cities=city_name_list[13:])


print('**************************************************')
print('Cityscapes clear')

def get_cityscapes_clear_train_dicts():
    return get_cityscapes_dicts(data_cityscapes_train_file_list, data_dir_cityscapes, data_dir_rain_addition, anno_dir_cityscapes, clear=True, rain=False, levels=['H', 'M', 'S'])

def get_cityscapes_clear_val_dicts():
    return get_cityscapes_dicts(data_cityscapes_val_file_list, data_dir_cityscapes, data_dir_rain_addition, anno_dir_cityscapes, clear=True, rain=False, levels=['H', 'M', 'S'])

print(len(get_cityscapes_clear_train_dicts())) # 2276
print(len(get_cityscapes_clear_val_dicts()))   # 699

print('\n**************************************************')
print('Cityscapes rain')

def get_cityscapes_rain_train_dicts():
    return get_cityscapes_dicts(data_cityscapes_train_file_list, data_dir_cityscapes, data_dir_rain_addition, anno_dir_cityscapes, clear=False, rain=True, levels=['H', 'M', 'S'])

def get_cityscapes_rain_val_dicts():
    # return get_cityscapes_dicts(data_cityscapes_val_file_list, data_dir_cityscapes, data_dir_rain_addition, anno_dir_cityscapes, clear=False, rain=True, levels=['H', 'M', 'S'])
    return get_cityscapes_dicts(data_cityscapes_val_file_list, data_dir_cityscapes, data_dir_rain_addition, anno_dir_cityscapes, clear=False, rain=True, levels=EVAL_LEVEL_LIST)

print(len(get_cityscapes_rain_train_dicts())) # 2276
print(len(get_cityscapes_rain_val_dicts()))   # 699

print('\n**************************************************')
print('Cityscapes all')

def get_cityscapes_all_train_dicts():
    return get_cityscapes_dicts(data_cityscapes_train_file_list, data_dir_cityscapes, data_dir_rain_addition, anno_dir_cityscapes, clear=True, rain=True, levels=['H', 'M', 'S'])

def get_cityscapes_all_val_dicts():
    return get_cityscapes_dicts(data_cityscapes_val_file_list, data_dir_cityscapes, data_dir_rain_addition, anno_dir_cityscapes, clear=True, rain=True, levels=['H', 'M', 'S'])

print(len(get_cityscapes_all_train_dicts())) # 2276*2 = 4552
print(len(get_cityscapes_all_val_dicts()))   # 699*2  = 1398

print('**************************************************')
print('Combined clear')

def get_combined_clear_train_dicts():
    return get_carla_clear_train_dicts() + get_cityscapes_clear_train_dicts()

def get_combined_clear_val_dicts():
    return get_carla_clear_val_dicts() + get_cityscapes_clear_val_dicts()

print(len(get_combined_clear_train_dicts())) # 2990 + 2276 = 5266
print(len(get_combined_clear_val_dicts()))   # 1173 + 699  = 1872

print('\n**************************************************')
print('Combined rain')

def get_combined_rain_train_dicts():
    return get_carla_rain_train_dicts() + get_cityscapes_rain_train_dicts()

def get_combined_rain_val_dicts():
    return get_carla_rain_val_dicts() + get_cityscapes_rain_val_dicts()

print(len(get_combined_rain_train_dicts())) # 2990 + 2276 = 5266
print(len(get_combined_rain_val_dicts()))   # 1173 + 699  = 1872

print('\n**************************************************')
print('Combined all')

def get_combined_all_train_dicts():
    return get_carla_all_train_dicts() + get_cityscapes_all_train_dicts()

def get_combined_all_val_dicts():
    return get_carla_all_val_dicts()[:1000] + get_cityscapes_all_val_dicts()[:1000]

print(len(get_combined_all_train_dicts())) # 5266*2 = 10532
print(len(get_combined_all_val_dicts()))   # 1872*2 = 3744

print('\n**************************************************')
print('Combined all night')

def get_combined_all_night_train_dicts():
    return get_combined_all_train_dicts() + get_carla_night_all_train_dicts()

def get_combined_all_night_val_dicts():
    return get_combined_all_val_dicts() + get_carla_night_all_val_dicts()[:1000]

print(len(get_combined_all_night_train_dicts())) # 10532 + 5684 = 16216
print(len(get_combined_all_night_val_dicts()))   # 3744  + 5948 = 9692

DatasetCatalog.clear()

print('**************************************************')

print('Carla clear dataset')
carla_clear_dataset = Detectron2CustomDataset('carla_clear_train', 'carla_clear_val', get_carla_clear_train_dicts, get_carla_clear_val_dicts)
# carla_clear_dataset.visualize_dataset(num_samples=4, size=(20, 10), show_original=True)
            
print('\nCarla rain dataset')
carla_rain_dataset = Detectron2CustomDataset('carla_rain_train', 'carla_rain_val', get_carla_rain_train_dicts, get_carla_rain_val_dicts)
# carla_rain_dataset.visualize_dataset(num_samples=1, size=(20, 10), show_original=True)

# print('\nCarla derained dataset')
# carla_derained_dataset = Detectron2CustomDataset('carla_derained_train', 'carla_derained_val', get_carla_rain_train_dicts, get_carla_derained_val_dicts)
# # carla_rain_dataset.visualize_dataset(num_samples=1, size=(20, 10), show_original=True)

##################################################

print('\n**************************************************')

print('Carla night clear dataset')
carla_night_clear_dataset = Detectron2CustomDataset('carla_night_clear_train', 'carla_night_clear_val', get_carla_night_clear_train_dicts, get_carla_night_clear_val_dicts)
# carla_night_clear_dataset.visualize_dataset(num_samples=1, size=(20, 10), show_original=True)
            
print('\nCarla night rain dataset')
carla_night_rain_dataset = Detectron2CustomDataset('carla_night_rain_train', 'carla_night_rain_val', get_carla_night_rain_train_dicts, get_carla_night_rain_val_dicts)
# carla_night_rain_dataset.visualize_dataset(num_samples=1, size=(20, 10), show_original=True)

##################################################

print('\n**************************************************')

print('Cityscapes clear dataset')
cityscapes_clear_dataset = Detectron2CustomDataset('cityscapes_clear_train', 'cityscapes_clear_val', get_cityscapes_clear_train_dicts, get_cityscapes_clear_val_dicts)
# cityscapes_clear_dataset.visualize_dataset(num_samples=10, size=(20, 10), show_original=True)

print('\nCityscapes rain dataset')
cityscapes_rain_dataset = Detectron2CustomDataset('cityscapes_rain_train', 'cityscapes_rain_val', get_cityscapes_rain_train_dicts, get_cityscapes_rain_val_dicts)
# cityscapes_rain_dataset.visualize_dataset(num_samples=1, size=(20, 10), show_original=True)

##################################################

print('\n**************************************************')
            
print('Combined clear dataset')
combined_clear_dataset = Detectron2CustomDataset('combined_clear_train', 'combined_clear_val', get_combined_clear_train_dicts, get_combined_clear_val_dicts)
# combined_clear_dataset.visualize_dataset(num_samples=1, size=(20, 10), show_original=True)
            
print('\nCombined rain dataset')
combined_rain_dataset = Detectron2CustomDataset('combined_rain_train', 'combined_rain_val', get_combined_rain_train_dicts, get_combined_rain_val_dicts)
# combined_rain_dataset.visualize_dataset(num_samples=1, size=(20, 10), show_original=True)
            
print('\nCombined all dataset')
combined_all_dataset = Detectron2CustomDataset('combined_all_train', 'combined_all_val', get_combined_all_train_dicts, get_combined_all_val_dicts)
# combined_all_dataset.visualize_dataset(num_samples=1, size=(20, 10), show_original=True)

##################################################

print('\n**************************************************')
            
print('Combined all night dataset')
combined_all_night_dataset = Detectron2CustomDataset('combined_all_night_train', 'combined_all_night_val', get_combined_all_night_train_dicts, get_combined_all_night_val_dicts)
# combined_all_dataset.visualize_dataset(num_samples=20, size=(20, 10), show_original=True)


trainer_clear = Detectron2Trainer('combined_all_train', 'combined_all_val', output_folder='./output_combined_all_40k')
trainer_clear.load()
trainer_clear.train()

torch.cuda.empty_cache()
