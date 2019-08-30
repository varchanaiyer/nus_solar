import argparse
import os
import time
import glob
import numpy as np
from tqdm import tqdm
import preprocess

import keras
from keras.models import Model, load_model
from keras.applications import mobilenet, MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.utils import multi_gpu_model
from keras.preprocessing import image

from keras.layers import DepthwiseConv2D, ReLU
relu6 = ReLU(6.)

from keras.utils.generic_utils import CustomObjectScope
#import ipdb; ipdb.set_trace()

#with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6, 'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
#    model = load_model(args.model)

def get_data_paths(folder):

    paths = glob.glob(os.path.join(folder, '*.jpg'))

    return np.asarray(paths)


def get_image(path):

    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return np.asarray(x)


def benchmark(args):

    if not args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    pp_time = []
    p_time = []
    t_time = []

    train_paths, train_l, train_i, test_paths, test_l, test_i, valid_paths, valid_l, valid_i = preprocess.get_data_paths_labels()

    load_time = time.time()
    from keras.utils.generic_utils import CustomObjectScope
    #import ipdb; ipdb.set_trace()

    model = load_model(args.model, custom_objects={'relu6': relu6})

    print(f'Time taken to load model is : {time.time()-load_time}')
    print('Sleeping')
    time.sleep(5)

    if args.multi:
        model = multi_gpu_model(model, gpus=3)

    for _ in tqdm(range(int(args.i))):
        for path, ird  in tqdm(zip(valid_paths, valid_i)):
            total_time = time.time()
            preprocess_time = time.time()
            img = get_image(path)
            pp_time.append(time.time() - preprocess_time)
            ird=np.reshape(ird, (-1, 1))

            prediction_time = time.time()
            model.predict({'input_1': img, 'ird':ird})
            p_time.append(time.time() - prediction_time)

            t_time.append(time.time() - total_time)

    if not os.path.exists(args.loc):
        os.makedirs(os.path.join(f'{args.loc}'))
        print(f'Saving to dir {args.loc}')

    np.save(os.path.join(f'{args.loc}', f'pp_time'), pp_time)
    np.save(os.path.join(f'{args.loc}', f'p_time'), p_time)
    np.save(os.path.join(f'{args.loc}', f't_time'), t_time)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument(
        '--loc',
        help='The path where you want to save the benchamrking results')
    a.add_argument('--model', help='The path of the saved model file')
    a.add_argument('--gpu', action='store_true', help='To use GPU')
    a.add_argument('--multi', action='store_true', help='To use multi GPU')
    a.add_argument(
        '--images', help='Location of test images')
    a.add_argument(
        '--i',
        default=1,
        help="Test images will be iterated i'th many times. Default = 1")
    args = a.parse_args()

    benchmark(args)
