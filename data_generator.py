
# calculate_bounding_box and calculate_bounding_box complete code here git clone https://github.com/hasannasirkhan/BrainTumor-ROI-Crop.git
import gzip
import keras
import shutil
import numpy as np
import os
import re
import  nibabel as nib
import tensorflow as tf


def extract_gzipped_file(gz_file, extract_file):
    with gzip.open(gz_file, 'rb') as f_in:
        with open(extract_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            
def calculate_bounding_box(all_ids,TRAIN_DATASET_PATH):
    # Initialize min and max indices to large and small values respectively
    min_indices = np.array([float('inf'), float('inf'), float('inf')])
    max_indices = np.array([0, 0, 0])
    
    len_ = len(all_ids)
    # Loop through the list of segmentation volumes
    for idx,i in enumerate(all_ids):
        #specify the path to the file 
        case_path = os.path.join(TRAIN_DATASET_PATH, i)
        
        #we need first to unzip the file
        extract_gzipped_file(os.path.join(case_path,f'{i}_seg.nii.gz'),
                 'BraTS2021_seg.nii')
        
        # Load the segmentation mask
        mask_volume = nib.load('BraTS2021_seg.nii').get_fdata()

        # Find the indices of non-zero values in the mask
        non_zero_indices = np.argwhere(mask_volume > 0)

        # Update the min and max indices
        min_indices = np.minimum(min_indices, np.min(non_zero_indices, axis=0))
        max_indices = np.maximum(max_indices, np.max(non_zero_indices, axis=0))
        
        #the last part is just to show the progress of getting bounding_box
        pattern = '[0-9]+'
        res = re.findall(pattern,i)
        if len(res)>0:
            print(f'done with :{res[1]} , it left {len_ - (idx+1)}',end='\r')
            
    # Convert indices to integers
    min_indices = min_indices.astype(int)
    max_indices = max_indices.astype(int)
    
    print('done calculate bounding box')
    print("Bounding Box Min Indices:", min_indices)
    print("Bounding Box Max Indices:", max_indices)
    print("Bounding Box Dimensions:", max_indices - min_indices + 1)
    
    return min_indices, max_indices

def crop_volumes(x,min_indices,max_indices):
    
    x = x[min_indices[0]:max_indices[0] + 1,
                                   min_indices[1]:max_indices[1] + 1,
                                   min_indices[2]:max_indices[2] + 1]
    return x 

#original code https://www.kaggle.com/code/rastislav/3d-mri-brain-tumor-segmentation-u-net
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,min_indices,max_indices,DATASET_PATH,VOLUME_SLICES,VOLUME_START_AT,dim, list_IDs,batch_size=1,n_classes=4, n_channels = 4,preprocess_input=None, shuffle=True,args=None):
        'Initialization'
        self.preprocess_input = preprocess_input
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.VOLUME_SLICES = VOLUME_SLICES
        self.VOLUME_START_AT = VOLUME_START_AT
        self.dim = dim
        self.DATASET_PATH = DATASET_PATH
        self.min_indices = min_indices
        self.max_indices = max_indices

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        Batch_ids = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(Batch_ids)
        return tf.convert_to_tensor(X), y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        # Initialization
        X = np.zeros((self.batch_size*self.VOLUME_SLICES, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size*self.VOLUME_SLICES, *self.dim))

        #flip the volum
        for c,i in enumerate(Batch_ids):
            #defined the directory
            case_path = os.path.join(self.DATASET_PATH, i)
            #unzip the modalities
            extract_gzipped_file(os.path.join(case_path,f'{i}_t2.nii.gz'),
                     'BraTS2021_t2.nii')
            extract_gzipped_file(os.path.join(case_path,f'{i}_t1ce.nii.gz'),
                     'BraTS2021_t1ce.nii')
            extract_gzipped_file(os.path.join(case_path,f'{i}_t1.nii.gz'),
                     'BraTS2021_t1.nii')
            extract_gzipped_file(os.path.join(case_path,f'{i}_flair.nii.gz'),
                     'BraTS2021_flair.nii')
            extract_gzipped_file(os.path.join(case_path,f'{i}_seg.nii.gz'),
                     'BraTS2021_seg.nii')
            #load the data
            t2= nib.load('BraTS2021_t2.nii').get_fdata()
            ce =  nib.load('BraTS2021_t1ce.nii').get_fdata()
            t1 = nib.load('BraTS2021_t1.nii').get_fdata()
            flair = nib.load('BraTS2021_flair.nii').get_fdata()
            seg = nib.load('BraTS2021_seg.nii').get_fdata()
            #preprocesing
            if self.preprocess_input !=None:
                t2    = self.preprocess_input(t2,self.min_indices,self.max_indices)
                ce    = self.preprocess_input(ce,self.min_indices,self.max_indices)
                t1    = self.preprocess_input(t1,self.min_indices,self.max_indices)
                flair = self.preprocess_input(flair,self.min_indices,self.max_indices)
                seg   = self.preprocess_input(seg,self.min_indices,self.max_indices)
            
            #loop over slices to flip the slices from axis 2 to axis 0
            for j in range(self.VOLUME_SLICES):
                X[j +self.VOLUME_SLICES*c,:,:,0] = flair[:,:,j+self.VOLUME_START_AT];
                X[j +self.VOLUME_SLICES*c,:,:,1] =    t1[:,:,j+self.VOLUME_START_AT];
                X[j +self.VOLUME_SLICES*c,:,:,2] =    ce[:,:,j+self.VOLUME_START_AT];
                X[j +self.VOLUME_SLICES*c,:,:,3] =    t2[:,:,j+self.VOLUME_START_AT];

                y[j +self.VOLUME_SLICES*c] = seg[:,:,j+self.VOLUME_START_AT];

        # Generate masks   
        y[y==4] = 3        #(0,1,2,4) -> (0,1,2,3)
        y = tf.one_hot(y, 4)
        return X,y