from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import json
import numpy as np
import argparse
import time
import tensorflow as tf
slim = tf.contrib.slim
from dataloader import MDEdataloader
import network
import cv2
from metrics import compute_metrics_for_single_map


parser = argparse.ArgumentParser(description="Evaluation code for nyu_depth_v2 of "
                                 "Regression for Depth for Monocular Depth Prediction.")
parser.add_argument("--output_stride",
                    type=int,
                    default=16,
                    help="output stride for testing.")
########################################################################################################################
#  checkpoint path to restore trained model from, and the process id of the model
parser.add_argument("--checkpoint_path",
                    type=str,
                    default="./checkpoint/",
                    help="path to a specific checkpoint to load.")
parser.add_argument("--process_id_for_evaluation",
                    type=int,
                    default=12582,
                    help="process_id_of_the_trained_model.")

test_args = parser.parse_args()

test_file = "./filenames/nyu_depth_v2_test_654.txt"
dataset = "nyu_depth_v2"

# eigen's crop for evaluation
crop = [-460, -20, 25, 617]

process_str_id = str(test_args.process_id_for_evaluation)
CKPT_PATH = os.path.join(test_args.checkpoint_path, dataset, process_str_id)
result_path = os.path.join(CKPT_PATH, 'result/')
if not os.path.exists(result_path):
    print("CheckPoint folder:", result_path)
    os.makedirs(result_path)
    os.makedirs(result_path+'/preds')

with open(CKPT_PATH+'/args'+process_str_id+'.json', 'r') as fp:
    train_args = json.load(fp)


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

train_args = Dotdict(train_args)

train_args.aspp_rates = list(map(lambda r: int(r*train_args.output_stride/test_args.output_stride),
                                 train_args.aspp_rates))
train_args.output_stride = test_args.output_stride


with tf.Graph().as_default():
    # Dataloader
    dataloader = MDEdataloader(dataset,
                               1, 1, 1,
                               test_file=test_file)

    test_dataset = dataloader.test_dataset
    print("total number of test samples: {}".format(dataloader.num_test_samples))

    iterator = test_dataset.make_one_shot_iterator()
    image_ts, depth_ts, im_size_ts = iterator.get_next()

    logits_ts = network.mdr_net(image_ts,
                                train_args,
                                [dataloader.INPUT_HEIGHT, dataloader.INPUT_WIDTH],
                                is_training=False,
                                reuse=False)

    pd_normlized_depth_ts = tf.nn.sigmoid(logits_ts)
    pd_depth_map_ts = dataloader.MAX_DEPTH * pd_normlized_depth_ts

    variables_to_restore = slim.get_variables_to_restore()
    restorer = tf.train.Saver(variables_to_restore)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(CKPT_PATH + '/')
        if ckpt and ckpt.model_checkpoint_path:
            restorer.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise NameError("Trained model with process id " + process_str_id + "may not exist")
        print("Model checkpoits for process id " + process_str_id + " restored!")

        times = np.zeros(dataloader.num_test_samples, np.float32)

        ABS_REL = np.zeros(dataloader.num_test_samples, np.float32)
        SQ_REL = np.zeros(dataloader.num_test_samples, np.float32)
        RMSE = np.zeros(dataloader.num_test_samples, np.float32)
        RMSE_log = np.zeros(dataloader.num_test_samples, np.float32)
        Log10 = np.zeros(dataloader.num_test_samples, np.float32)
        ACC1 = np.zeros(dataloader.num_test_samples, np.float32)
        ACC2 = np.zeros(dataloader.num_test_samples, np.float32)
        ACC3 = np.zeros(dataloader.num_test_samples, np.float32)

        for step in range(0, dataloader.num_test_samples):
            start_time = time.time()
            gt_depth, im_size, depth_map = sess.run([depth_ts,
                                                     im_size_ts,
                                                     pd_depth_map_ts])
            times[step] = time.time() - start_time


            im_size = im_size[0]
            resized_depth_map = cv2.resize(depth_map[0],
                                           (im_size[1], im_size[0]),
                                           interpolation=cv2.INTER_LINEAR)
            saved_depth_map = resized_depth_map*256
            saved_depth_map = saved_depth_map.astype(np.uint16)
            cv2.imwrite(result_path + '/preds/pred_' + '%03d'%(step + 1) + '.png', saved_depth_map)


            ABS_REL[step], SQ_REL[step], RMSE[step], RMSE_log[step], \
            Log10[step], ACC1[step], ACC2[step], ACC3[step] \
                = compute_metrics_for_single_map(gt_depth[0, crop[0]:crop[1], crop[2]:crop[3], 0],
                                                 resized_depth_map[crop[0]:crop[1], crop[2]:crop[3]],
                                                 cap=10)

        print_string = 's/example: {:2.4f} \n |abs_rel: {:2.3f} |sq_rel: {:2.3f} ' \
                       '|rmse: {:2.3f} |rmse_log: {:2.3f} |log10: {:2.3f} ' \
                       '|acc1: {:.3f} |acc2: {:.3f} |acc3: {:.3f}'
        print(print_string.format(times.mean(),
                                  ABS_REL.mean(),
                                  SQ_REL.mean(),
                                  RMSE.mean(),
                                  RMSE_log.mean(),
                                  Log10.mean(),
                                  ACC1.mean(),
                                  ACC2.mean(),
                                  ACC3.mean()))

        f = open(result_path+'metrics.txt', 'w+')
        print(print_string.format(times.mean(),
                                  ABS_REL.mean(),
                                  SQ_REL.mean(),
                                  RMSE.mean(),
                                  RMSE_log.mean(),
                                  Log10.mean(),
                                  ACC1.mean(),
                                  ACC2.mean(),
                                  ACC3.mean()), file=f)
        f.close()


