import argparse
import os
import random
import re
from math import cos, pi

import cv2
import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import (LearningRateScheduler, ModelCheckpoint,
                             ReduceLROnPlateau, TensorBoard)
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

import constants
from generator import segmentationGenerator
from helpers import make_overlay
from loss import modelLoss
from model import create_Model


'''
TODO 
re-evaluate loss function 
overlay image a bit better
resnet old decoder is broken so should be removed
evaluation metrics for generated output

- (no longer needed) linear histogram equilization
'''

# Default parameters
model_epoch_number = 38#16
resnet_type = 50
session_id = 'seg_UNet_se_d_2021-05-28 11 33 02_batchsize_4_resnet_50_se8'
batchSize = 1
model_epoch_base = '_weights_epoch'
output_img_base_dir = 'output'
model_base_dir = 'models'
visualize_default = False
use_test_images_default = True

# Argparser
argparser = argparse.ArgumentParser(description='Testing')

argparser.add_argument('-e',
                       '--epoch',
                       default=model_epoch_number,
                       help='model epoch number')
argparser.add_argument('-s',
                       '--session',
                       default=session_id,
                       help='session id number')
argparser.add_argument('-b',
                       '--batch',
                       default=batchSize,
                       help='batch size')
argparser.add_argument('-r',
                       '--resnet',
                       default=resnet_type,
                       help='resnet type')
argparser.add_argument('-v',
                       '--visualize',
                       default=visualize_default,
                       help='enable visualize')
argparser.add_argument('-t',
                       '--test',
                       default=use_test_images_default,
                       help='use test images')

args = argparser.parse_args()

# convert string arguments to appropriate type
if type(args.visualize) is not bool:
    args.visualize = args.visualize == '1' or args.visualize.lower() == 'true'
args.epochs = int(args.epoch)
args.batch = int(args.batch)
args.resnet = constants.EncoderType(int(args.resnet))
if type(args.test) is not bool:
    args.test = args.test == '1' or args.test.lower() == 'true'

src_type = 'test' if args.test else 'train'
eval_image_input_path = constants.data_test_image_dir if args.test else constants.data_train_image_dir

output_img_path = os.path.join(output_img_base_dir, args.session, str(args.epoch), src_type)
model_path = os.path.join(model_base_dir, args.session)

def get_model_name_from_epoch(src, epoch):
    models = os.listdir(src)

    pattern = re.compile(model_epoch_base + "[a-zA-Z_]*[0]*" + str(args.epoch))

    for modelName in models:
        if pattern.search(modelName):
            return modelName
    
    return None


model_name = get_model_name_from_epoch(model_path, args.epoch)

if model_name is None:
    print("Cannot find model corresponding to model_path: '" + model_path + "' and epoch " + str(args.epoch))
    exit(1)

# build loss
lossClass = modelLoss(0.001,0.85,256,256,batchSize)
loss = lossClass.applyLoss 

# build model
model = create_Model(input_shape=(256,256,3), encoder_type=args.resnet)

model.compile(optimizer=Adam(lr=1e-3),loss=loss, metrics=['accuracy'])

def modelPredictWrapper(model, inputImg):
        #---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        #---------------------------------------------------#
#        old_img = copy.deepcopy(image)
#        orininal_h = np.array(image).shape[0]
#        orininal_w = np.array(image).shape[1]

        #---------------------------------------------------#
        #   进行不失真的resize，添加灰条，进行图像归一化
        #---------------------------------------------------#
        #img, nw, nh = self.letterbox_image(image,(self.model_image_size[1],self.model_image_size[0]))
        #img = np.asarray([np.array(img)/255])
        
        #---------------------------------------------------#
        #   图片传入网络进行预测
        #---------------------------------------------------#
        modelOutput = model.predict(np.expand_dims(inputImg,0))
        #---------------------------------------------------#
        #   取出每一个像素点的种类
        #---------------------------------------------------#
        modelOutput = modelOutput.argmax(axis=-1).reshape([constants.input_shape[0],constants.input_shape[1]])
        #--------------------------------------#
        #   将灰条部分截取掉
        #--------------------------------------#
        #pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]

        #------------------------------------------------#
        #   创建一副新图，并根据每个像素点的种类赋予颜色
        #------------------------------------------------#
        seg_img = np.zeros((np.shape(modelOutput)[0],np.shape(modelOutput)[1],3))
#        seg_img = np.zeros((320,320,3))
        for c in range(constants.num_classes):
            seg_img[:,:,0] += ((modelOutput[:,: ] == c )*( constants.colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((modelOutput[:,: ] == c )*( constants.colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((modelOutput[:,: ] == c )*( constants.colors[c][2] )).astype('uint8')

     

        return seg_img
'''
def modelPredictWrapper(model, inputImg):
    modelOutput = model.predict(np.expand_dims(inputImg,0))# * 640 * 0.3

#    if constants.use_unet:
#        # change from (1, 640, 192, 2) to (1, 640, 192, 3)
#        zeros = np.zeros([modelOutput.shape[0], modelOutput.shape[1], modelOutput.shape[2], 1])
#        modelOutput = np.concatenate((modelOutput, zeros), axis=3)
#        
#        # change order of channels
#        # modelOutput[:,:,:,[0,1,2]] = modelOutput[:,:,:,[1,2,0]]
    
    return modelOutput'''

def evaluateModel(model,batchSize, visualize):
#    val_generator  = segmentationGenerator(constants.data_train_image_dir,constants.data_train_gt_dir, batch_size=args.batch, shuffle=False, augmentations=False)
    # scores = model.evaluate_generator(val_generator, verbose=1)
    # print("Total Loss")
    # print(scores)
    ARD = 0
    count = 0 
    ABS = 0 
    SQR = 0 
    # Random Qualitative Evaluation
#    imageList = os.listdir(constants.data_test_image_dir)
#    if constants.system_files in imageList:
#        imageList.remove(constants.system_files)
#    imageName = random.choice(imageList)
#    inputImg = cv2.imread(constants.data_test_image_dir + imageName)
#    rawImage = cv2.resize(inputImg, (320,320))
#    inputImgOrig  = cv2.resize(cv2.imread(constants.data_test_image_dir + imageName), (320,320))
#    inputImg  = np.transpose(inputImgOrig.astype('float32'), axes=[1,0,2])
#    output = modelPredictWrapper(model, inputImg)
    def displayOutput(output, dim=0):
        output = np.squeeze(output)
        outputTransformed = np.transpose(  output,    axes=[1,0,2])
#        outputTransformed= output
        # outputTransformed = outputTransformed - np.mean(outputTransformed)
        # outputTransformed = np.clip(outputTransformed, (np.mean(outputTransformed) - 2*np.std(outputTransformed)), (np.mean(outputTransformed) + 2*np.std(outputTransformed)))
        # outputTransformed = outputTransformed - np.min(outputTransformed)
        outputTransformed = np.clip(outputTransformed / np.max(outputTransformed) * 255, 0, 255).astype('uint8')
        return outputTransformed

    def generateHist(output, name, dim=0):
        outputSqueezed = output.squeeze()

        outputChannelSingle = outputSqueezed[:,:,dim]
        _ = plt.hist(outputChannelSingle, bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram with 'auto' bins")

        plt.show()
        plt.savefig(name)
        plt.close()

#    outputDisplay = displayOutput(output, 2)
#
#
#
#    overlayedImage = cv2.addWeighted(inputImgOrig, 0.8, outputDisplay, 0.2, 0)

    if visualize:
        cv2.imshow("Input Image", rawImage)
        cv2.imshow("Segmentation Prediction",  outputDisplay)
        cv2.imshow("Segmentation Overlay",  overlayedImage)


        #cv2.imwrite("../Images/InputImages.png",  rawImage )
        #cv2.imwrite("../Images/SegmentationPrediction.png",  outputDisplay. )
        #cv2.waitKey(-1)


    # actual Evaluation
    imgs = os.listdir(eval_image_input_path)
    print("")
    for filename in os.listdir(eval_image_input_path):
        if filename == '.DS_Store':
            continue
        inputImgOrig    = cv2.resize(cv2.imread(os.path.join(eval_image_input_path, filename)), (256,256))
        inputImg    = np.transpose(inputImgOrig.astype('float32'),      axes=[1,0,2])
#        inputImg= inputImgOrig
        output = modelPredictWrapper(model, inputImg)
        count += 1
        outputTransformed = displayOutput(output)
        #顺时针旋转90，水平翻转
        outputTransformed=cv2.flip(cv2.rotate(outputTransformed,cv2.ROTATE_90_CLOCKWISE),1)
#        overlayedImage = cv2.addWeighted(inputImgOrig, 0.8, outputTransformed, 0.2, 0)

        cv2.imwrite(os.path.join(output_img_path, filename),  outputTransformed )
#        cv2.imwrite(os.path.join(output_img_path, "overlay_" + filename),  overlayedImage )

        # generateHist(output, os.path.join(output_img_path, "overlay_" + filename[:-4] + "_hist_0" + filename[-4:]),0)
        # generateHist(output, os.path.join(output_img_path, "overlay_" + filename[:-4] + "_hist_1" + filename[-4:]),1)
        # generateHist(output, os.path.join(output_img_path, "overlay_" + filename[:-4] + "_hist_2" + filename[-4:]),2)

        print(count, " of ", len(imgs) , end='\r')
    print("Mean ARD: ", ARD / count)
    print("Mean SQR: ", SQR / count)
    print("")

print("\n\n")
print("Model Session ID: {}".format(args.session))
print("Model from epoch: {}".format(args.epoch))
print("Testing model: {}".format(model_name))
if not os.path.exists(output_img_path):
    os.makedirs(output_img_path)
print("Reading dataset input from: {}".format(eval_image_input_path))
print("Writing model output to: {}".format(output_img_path))

model.load_weights(os.path.join(model_path, model_name))
evaluateModel(model,args.batch, args.visualize)

print("Completed. Model outputs can be found at: {}".format(output_img_path))