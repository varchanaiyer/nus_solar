import argparse
import os
import time
import glob
import numpy as np
from tqdm import tqdm
import preprocess
import tensorflow as tf

def benchmark(args):

    if not args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    pp_time = []
    p_time = []
    t_time = []

    train_paths, train_l, train_i, test_paths, test_l, test_i, valid_paths, valid_l, valid_i = preprocess.get_data_paths_labels()
    import ipdb; ipdb.set_trace()
    load_time = time.time()
    sess=tf.Session()
    init=tf.global_variables_initializer()

    sess.run(init)

    saver = tf.train.import_meta_graph("models/model1/model3.meta")
    saver.restore(sess, 'models/model1/model3')

    graph=tf.get_default_graph()

    x_input=graph.get_tensor_by_name('x:0')
    ird_input=graph.get_tensor_by_name('ird:0')
    y_output=graph.get_tensor_by_name('dense_3/BiasAdd:0')

    print(f'Time taken to load model is : {time.time()-load_time}')
    print('Sleeping')
    time.sleep(5)

    if args.multi:
        model = multi_gpu_model(model, gpus=3)

    for _ in tqdm(range(int(args.i))):
        for path, ird  in tqdm(zip(valid_paths, valid_i)):
            total_time = time.time()
            preprocess_time = time.time()
            img=np.asarray(preprocess.reshape_data([path]))
            pp_time.append(time.time() - preprocess_time)
            ird=np.reshape(ird, (-1, 1))

            prediction_time = time.time()
            feed_dict=({x_input: img, ird_input:ird})
            p_time.append(time.time() - prediction_time)

            sess.run([y_output] ,feed_dict=feed_dict)
            #y_output.eval(session=sess, feed_dict=feed_dict)

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
