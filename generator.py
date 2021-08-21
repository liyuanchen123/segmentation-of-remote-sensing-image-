import os
import random

import cv2
import keras
import numpy as np

import constants
import dataset


# TODO use imgaug for more robust image augmentation
def preprocess_input(image, randomVals):
    '''
    performs data augmentations listed below each with chance == 50%
    - Horiontal flip
    - Random brightness +- 0.2
    - Random contrast +- 0.2
    - Random saturation +- 0.2
    - Hue Jitter +- 0.1

    all random values provided in range (0,1)
    '''
    if randomVals[0] > 0.5:
        # flip image horizontally
        #image = np.flip(image, 1)
        None
    if randomVals[1] > 0.5:
        # increase/ decrease contrast
        image = np.uint8(np.clip(image * (0.8 + randomVals[2]/2.5), a_min=0, a_max=255))
    # Convert image to HSV for some transformations
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int32)
    if randomVals[3] > 0.5:
        # change brightness of image
        hsv_image[:,:,2] += int(((randomVals[4]/2.5 )- 0.2 ) * 255.) 
        hsv_image[:,:,2] = np.clip(hsv_image[:,:,2], a_min =0, a_max = 255) 
    if randomVals[5] > 0.5:
        # change staturation
        hsv_image[:,:,1] += int(((randomVals[6]/2.5 )- 0.2 ) * 255.)
        hsv_image[:,:,1] = np.clip(hsv_image[:,:,1], a_min = 0, a_max = 255) 
    if randomVals[7] > 0.5:
        # change Hue
        hsv_image[:,:,0] += int(((randomVals[8]/2.5 )- 0.2 ) * 179.)
        hsv_image[:,:,0] = np.clip(hsv_image[:,:,0], a_min = 0, a_max = 179) 
    # Convert image back from HSV
    image = cv2.cvtColor(np.uint8(hsv_image), cv2.COLOR_HSV2BGR)
    return image


    #for x in range(0,9):
    #    randomVals.append(random.random())
    #input_img = preprocess_input(image=input_img, randomVals=randomVals)

def adjustData(seg):
    
        #img = img/255.
        seg=cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
        flag= np.zeros((constants.input_shape[0],constants.input_shape[1]))
        seg = seg/255.0
        seg[seg<0.5]= 0
        seg[seg>=0.5]= 1
        #print(seg)
                
        for m in range(constants.input_shape[0]):
            for k in range(constants.input_shape[1]):
                if (seg[m,k,0]==1 and seg[m,k,1]==1 and seg[m,k,2]==1): flag[m,k]= 0
                elif (seg[m,k,0]==0 and seg[m,k,1]==0 and seg[m,k,2]==1): flag[m,k]=1
                elif (seg[m,k,0]==0 and seg[m,k,1]==1 and seg[m,k,2]==1): flag[m,k]=2
                elif (seg[m,k,0]==0 and seg[m,k,1]==1 and seg[m,k,2]==0): flag[m,k]=3
                elif (seg[m,k,0]==1 and seg[m,k,1]==1 and seg[m,k,2]==0): flag[m,k]=4
                elif (seg[m,k,0]==1 and seg[m,k,1]==0 and seg[m,k,2]==0): flag[m,k]=5
        
        label = seg[:,:,0]
        
        # print('-----1-----', img.shape, label.shape)
        new_label = np.zeros(label.shape+(constants.num_classes,))
        
        # print('new_label.shape',new_label.shape)
        for i in range(constants.num_classes):
            new_label[flag==i,i] = 1
        label = new_label
   
        return (label)


class segmentationGenerator(keras.utils.Sequence):
    img_list = []
    '''Generates data for Keras'''
    '''Framework taken from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly'''
    '''Provided directories should contain the same number of files all with the same names to their pair image'''
    def __init__(self, img_dir, seg_dir, batch_size = 64, image_size=(256,256), shuffle=True, augmentations=True, test=False):
        # download dataset if not exist
        #dataset.verify_dataset()

        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentations = augmentations
        self.inputs = []
        self.test = test
        self.initalSetup()


    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.inputs) / self.batch_size))


    def __getitem__(self, index):
        '''Generate one batch of data'''
        outX  =  np.empty((self.batch_size,  *self.image_size, 3))
        if constants.use_unet:
            outY  =  np.empty((self.batch_size,  *self.image_size, 6))
        else:
            outY  =  np.empty((self.batch_size,  *self.image_size, 3))
        outY_0 = np.empty((self.batch_size, *self.image_size, 3))
        outY_1 = np.empty((self.batch_size, *self.image_size, 3))

        imageNames = self.inputs[index*self.batch_size:(index+1)*self.batch_size]
        
#        inputs = []
#        targets = []

        for _, imageNameSet in enumerate(imageNames):
            img_path = os.path.join(self.img_dir, imageNameSet)   #目录和文件名合成一个路径
            seg_path = os.path.join(self.seg_dir, imageNameSet)

            img_orig = cv2.imread(img_path)
            seg_orig = cv2.imread(seg_path)
            #print(seg_path)
            
            # remove red from output
            #seg_road = seg_orig
            '''# seg_road[:,:,2] = np.zeros([seg_road.shape[0], seg_road.shape[1]])
            if constants.use_unet:
                seg_road[:,:,1] = 255-seg_road[:,:,0]
                seg_road[:,:,2] = np.zeros([seg_road.shape[0], seg_road.shape[1]])
            else:
                seg_road[:,:,2] = seg_road[:,:,0]
                seg_road[:,:,1] = seg_road[:,:,0]
#####################################'''
            if (seg_orig is None):
                print("Error in seg path: " + seg_path)
                continue

            img = cv2.resize(img_orig, dsize=self.image_size)
            seg = cv2.resize(seg_orig, dsize=self.image_size)
            
#            print(img.shape)
#            cv2.imshow('test', img)
#            cv2.waitKey(-1)
#            print(seg.shape)
#            cv2.imshow('test', seg)
#            cv2.waitKey(-1)
            
            if self.augmentations:
                randomVals = []
                for x in range(0,9):
                    randomVals.append(random.random())
                img_augmented = preprocess_input(image=img, randomVals=randomVals)
            else:
                img_augmented = img

            outX[_]   =  np.transpose(img_augmented, axes=[1,0,2])
            #outY_0[_] =  np.transpose(img,           axes=[1,0,2])
            #outY_1[_] =  np.transpose(seg,           axes=[1,0,2])
            #inputs.append(np.array(jpg)/255)
            outY[_] = adjustData(seg)
            # 从文件中读取图像
#            seg = np.array(seg)
#            seg[seg >= constants.num_classes] = constants.num_classes
#            
#            # 转化成one_hot的形式
#            seg_label = np.eye(constants.num_classes+1)[seg.reshape(-1)]
#            #print(outY[_])
#            seg_label = seg_label.reshape((320,320,constants.num_classes+1))
#            print(seg_label.shape)
#            targets.append(seg_labels)
#            
#            if len(targets) == self.batch_size:
#                tmp_inp = np.array(inputs)
#                tmp_targets = np.array(targets)
#                inputs = []
#                targets = []
#                yield tmp_inp, tmp_targets

##########################################
#            if constants.use_unet:
#                two_seg = seg[:,:,0:2]
#                outY[_] = np.transpose(two_seg, axes=[1,0,2])
#                zeros = np.zeros([outY[_].shape[1], outY[_].shape[0],1])
#                print(zeros.shape)
#                m = np.concatenate((outY[_], zeros), axis=2)
                
#            else:
#                outY[_] =  np.transpose(seg,           axes=[1,0,2])

#            print(m.shape)
#            test_out = outY[_].astype('uint8')#np.transpose(left_augmented,     axes=[1,0,2])
#            cv2.imshow('test', test_out)
#            cv2.waitKey(-1)

            # outY = np.concatenate([outY_0,outY_1], axis=3)
        

        return outX, outY #[outY_0, outY_1, outY_2, outY_3]
                        

    def on_epoch_end(self):
        ''' Shuffle the data if that is required'''
        if self.shuffle:
            random.shuffle(self.inputs)   
    
    def initalSetup(self):
        #print("")
        if (len(segmentationGenerator.img_list) == 0):
            imgs = os.listdir(self.img_dir)
            imgs.sort()

            systemFiles = '.DS_Store'
            
            prefixes = ('.')
            for word in imgs[:]: 
                if word.startswith(prefixes) or word is systemFiles:
                    imgs.remove(word)

            if self.shuffle:
                random.shuffle(imgs)
            
            segmentationGenerator.img_list = imgs
        
        if (self.test):
            self.inputs = segmentationGenerator.img_list[int(len(segmentationGenerator.img_list) * constants.train_ratio):]
        else:
            self.inputs = segmentationGenerator.img_list[0:int(len(segmentationGenerator.img_list) * constants.train_ratio)]

        #self.inputs = self.inputs[0:100]   
        print("")
        print("")



if __name__ == "__main__":
    train = segmentationGenerator(constants.data_train_image_dir, constants.data_train_gt_dir,   batch_size=8, shuffle=True)
    test = segmentationGenerator(constants.data_train_image_dir, constants.data_train_gt_dir,   batch_size=8, shuffle=True, test=True)

    train.__getitem__(1)
    test.__getitem__(1)

    print('Data generator test success.')
