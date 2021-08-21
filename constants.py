from enum import Enum

# ************************************************ #
# Parameters
input_shape = (256,256,3)
input_shape_full_size = (1024,320,3)#(1242, 374, 3)

num_classes = 6
#cv读取为BGR，此处也改为BGR
colors = [(255, 255, 255), (255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255)] 

#RGB形式
#    Impervious surfaces (RGB: 255, 255, 255)
#    Building (RGB: 0, 0, 255)
#    Low vegetation (RGB: 0, 255, 255)
#    Tree (RGB: 0, 255, 0)
#    Car (RGB: 255, 255, 0)
#    Clutter/background (RGB: 255, 0, 0)

road_color =        [255,0,255]
background_color =  [255,0,0]

system_files = '.DS_Store'

use_unet = True

train_ratio = 0.8 # 80% train, 20% test
assert train_ratio > 0 and train_ratio <= 1

# ************************************************ #
# Dataset Locations

data_train_image_dir = 'data/top1/split1'
data_train_gt_dir    = 'data/gt/split1'
data_test_image_dir  = 'data/top1/split'
data_location        = 'data'

# ************************************************ #
# Model Pre-Trained Weight Locations

resnet_18_model_path = "models/resnet18_imagenet_1000_no_top.h5"
resnet_50_model_path = "models/resnet50_imagenet_1000_no_top.h5"

def get_model_path(resnet_type):
    return "models/resnet{}_imagenet_1000_no_top.h5".format(str(resnet_type))

# ************************************************ #
# Model Types

class EncoderType(Enum):
    resnet18  = 18
    resnet34  = 34
    resnet50  = 50
    resnet101 = 101
    resnet152 = 152

classes         = 1000
dataset         = 'imagenet'
include_top     = False
models_location = "models"