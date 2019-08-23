import glob
import matplotlib.pyplot as plt
from random import shuffle
import re
import os
import numpy as np
from sklearn.model_selection import train_test_split

def get_file_path(base_path):
    '''
    returns all the file paths in the directory
    '''
    file_paths=glob.glob(os.path.join(base_path, '*.jpg'))

    return file_paths

def get_li(paths):
    '''
    returns the L&I value from the file path
    '''

    L=[]
    I=[]
    for path in paths:
        try:
            l = float(re.search(r'_L_(.*?)_I_', path).group(1))
            i = float(re.search(r'I_(.*?).jpg', path).group(1))
        except AttributeError:
            l=0
            i=0
        L.append(l)
        I.append(i)
    return L, I

def shuffle_and_split(paths):
    shuffle(paths)

    X_train, X_test = train_test_split(paths, test_size=0.20, random_state=42)
    X_train, X_valid = train_test_split(X_train, test_size=0.10, random_state=42)

    return X_train, X_test, X_valid

def reshape_data(data_paths):
    final_data=[]
    for each_path in data_paths:

        image=plt.imread(each_path)
        
        image=image[:, :, 0:3]

        img=image[55:430, 77:578]

        final_data.append(img)
        
    return final_data

    
def get_data_paths_labels():
    file_paths=get_file_path('../Solar_Panel_Soiling_Image_dataset/PanelImages')

    train_paths, test_paths, valid_paths=shuffle_and_split(file_paths)
    train_l, train_i = get_li(train_paths)
    test_l, test_i=get_li(test_paths)
    valid_l, valid_i=get_li(valid_paths)

    return train_paths, train_l, train_i, test_paths, test_l, test_i, valid_paths, valid_l, valid_i
    
