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
from utils import download_cnn_checkpoint_if_necessary, assign_to_device, average_gradients
import network
from losses import compute_depth_loss, compute_gradient_loss
from metrics import compute_metrics_for_multi_maps

parser = argparse.ArgumentParser(description="Monocular Depth Prediction Using Deeplab V3 +.")
########################################################################################################################
# dataset arguments
parser.add_argument("--dataset",
                    type=str,
                    default="nyu_depth_v2",
                    help="dataset to train on nyu_depth_v2.")
parser.add_argument("--train_file",
                    type=str,
                    default="./filenames/nyu_depth_v2_train_even.txt",
                    help="path to the train filenames text file.")
parser.add_argument("--val_file",
                    type=str,
                    default="./filenames/nyu_depth_v2_val_2k.txt",
                    help="path to the validation filenames text file.")

########################################################################################################################
# network settings
parser.add_argument("--cnn_model",
                    type=str,
                    default="resnet_v2_50",
                    choices=["resnet_v2_50", "resnet_v2_101", "vgg_16"],
                    help="CNN model to use as feature extractor. Choose one of resnet or vgg variants.")
parser.add_argument("--decoding_at_image_size",
                    action='store_true',
                    default=False,
                    help="Using original image in decoder.")
parser.add_argument("--output_stride",
                    type=int,
                    default=16,
                    help="Total output stride")
parser.add_argument("--multi_grid",
                    nargs="+",
                    type=int,
                    default=[1, 2, 4],
                    help="Multi grid atrous rates")
parser.add_argument("--aspp_rates",
                    nargs="+",
                    type=int,
                    default=[4, 8, 12],
                    help="Atrous Spatial Pyramid Pooling rates.")

########################################################################################################################
# regression loss argument
parser.add_argument("--loss_depth_norm",
                    type=str,
                    default="berhu",
                    help="norm for loss on depth, l1, l2 or berhu.")
parser.add_argument("--using_log_depth",
                    action='store_true',
                    default=False,
                    help="compute depth loss in log space.")

parser.add_argument("--loss_gradient_magnitude_norm",
                    type=str,
                    default="l1",
                    help="norm for loss on depth gradient magnitude, l1, l2 or berhu.")

parser.add_argument("--using_log_gradient_magnitude",
                    action='store_true',
                    default=False,
                    help="compute gradient magnitude loss in log space.")

parser.add_argument("--loss_gradient_magnitude_weight",
                    type=float,
                    default=0.0,
                    help="weight for loss on depth gradient magnitude.")

parser.add_argument("--loss_gradient_direction_norm",
                    type=str,
                    default="l1",
                    help="norm for loss on depth gradient direction, l1, l2 or berhu.")

parser.add_argument("--loss_gradient_direction_weight",
                    type=float,
                    default=0.0,
                    help="weight for loss on depth gradient direction.")

parser.add_argument("--loss_normal_weight",
                    type=float,
                    default=0.0,
                    help="weight for loss on depth normal.")


########################################################################################################################
# training hyperparameters
parser.add_argument("--batch_size",
                    type=int,
                    default=8,
                    help="batch size per GPU.")
parser.add_argument("--num_epochs",
                    type=int,
                    default=20,
                    help="number of epochs.")
parser.add_argument("--learning_rate",
                    type=float,
                    default=1e-4,
                    help="initial learning rate.")
parser.add_argument("--num_gpus",
                    type=int,
                    default=1,
                    help="number of GPUs to use for training.")
parser.add_argument("--num_threads",
                    type=int,
                    default=4,
                    help='number of threads to use for data loading.')
parser.add_argument("--batch_norm_epsilon",
                    type=float,
                    default=1e-5,
                    help="batch norm epsilon argument for batch normalization.")
parser.add_argument('--batch_norm_decay',
                    type=float,
                    default=0.9997,
                    help='batch norm decay argument for batch normalization.')
parser.add_argument("--l2_regularizer",
                    type=float,
                    default=1e-4,
                    help="l2 regularizer parameter.")

########################################################################################################################
# Imagenet-pretrained model checkpoint path, checkpoint path to save to and restore from, tensorboard log folder and
# process id for further model refining
parser.add_argument("--pretrained_ckpts_path",
                    type=str,
                    default="./nets/checkpoints/",
                    help="path to imagenet-pretrained resnet checkpoint to load.")
parser.add_argument("--checkpoint_path",
                    type=str,
                    default="./checkpoint/",
                    help="path to a specific checkpoint to load.")
parser.add_argument("--process_id_for_refining",
                    type=int,
                    default=None,
                    help='process_id_for_further_refining_the_model.')

args = parser.parse_args()



def main(_):
    process_str_id = str(args.process_id_for_refining if args.process_id_for_refining else os.getpid())
    CKPT_PATH = os.path.join(args.checkpoint_path, args.dataset, process_str_id)
    if not os.path.exists(CKPT_PATH):
        print("Checkpoint folder:", CKPT_PATH)
        os.makedirs(CKPT_PATH)
    with open(CKPT_PATH + '/args' + str(os.getpid()) +'.json', 'w') as fp:
        json.dump(args.__dict__, fp, sort_keys=True, indent=4)

    download_cnn_checkpoint_if_necessary(args.pretrained_ckpts_path, args.cnn_model)


    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # Dataloader
        dataloader = MDEdataloader(args.dataset,
                                   args.num_threads,
                                   args.batch_size * args.num_gpus,
                                   args.num_epochs,
                                   train_file=args.train_file,
                                   val_file=args.val_file)
        training_dataset = dataloader.train_dataset
        validation_dataset = dataloader.val_dataset
        print("total number of samples: {}".format(dataloader.num_train_samples))

        # A feedable iterator is defined by a handle placeholder and its structure. We could use the `output_types` and
        # `output_shapes` properties of either `training_dataset` or `validation_dataset` here, because they have
        # identical structure.
        handle = tf.placeholder(tf.string, shape=[])

        iterator = tf.data.Iterator.from_string_handle(handle,
                                                       training_dataset.output_types,
                                                       training_dataset.output_shapes)
        batch_images_ts, batch_depth_ts, max_depth_ts = iterator.get_next()

        # You can use feedable iterators with a variety of different kinds of iterator
        # (such as one-shot and initializable iterators).
        training_iterator = training_dataset.make_initializable_iterator()
        validation_iterator = validation_dataset.make_initializable_iterator()

        is_training_ts = tf.placeholder(tf.bool, shape=[])


        # OPTIMIZER
        steps_per_epoch = np.ceil(dataloader.num_train_samples/args.batch_size/args.num_gpus).astype(np.int32)
        num_total_steps = args.num_epochs * steps_per_epoch
        print("total number of steps: {}".format(num_total_steps))
        start_learning_rate = args.learning_rate

        with tf.variable_scope("optimizer_vars"):
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 10000, 0.9, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        tower_pd_depth_maps = []
        tower_total_losses = []
        tower_depth_losses = []
        tower_grad_magni_losses = []
        tower_grad_direc_losses = []
        tower_normal_losses = []

        tower_grads = []
        reuse_vars = False
        for i in range(args.num_gpus):
            #with tf.device('/gpu:%d' % i):
            with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):
                logits_ts = tf.cond(is_training_ts,
                                    true_fn=lambda: network.mdr_net(
                                        batch_images_ts[i * args.batch_size: (i + 1)*args.batch_size],
                                        args, [dataloader.INPUT_HEIGHT, dataloader.INPUT_WIDTH],
                                        is_training=True, reuse=reuse_vars),
                                    false_fn=lambda: network.mdr_net(
                                        batch_images_ts[i * args.batch_size: (i + 1)*args.batch_size],
                                        args, [dataloader.INPUT_HEIGHT, dataloader.INPUT_WIDTH],
                                        is_training=False,
                                        reuse=True))

                pd_normlized_depth_ts = tf.nn.sigmoid(logits_ts)
                pd_depth_maps = max_depth_ts[i * args.batch_size: (i + 1)*args.batch_size] * pd_normlized_depth_ts
                tower_pd_depth_maps.append(pd_depth_maps)

                loss_depth = compute_depth_loss(batch_depth_ts[i * args.batch_size: (i + 1)*args.batch_size],
                                                pd_depth_maps,
                                                args.loss_depth_norm,
                                                args.using_log_depth)
                tower_depth_losses.append(loss_depth)

                loss_grad_magni, loss_grad_direc, loss_normal = \
                    compute_gradient_loss(batch_depth_ts[i * args.batch_size: (i + 1) * args.batch_size],
                                          pd_depth_maps,
                                          args.using_log_gradient_magnitude,
                                          args.loss_gradient_magnitude_norm,
                                          args.loss_gradient_direction_norm)
                tower_grad_magni_losses.append(loss_grad_magni)
                tower_grad_direc_losses.append(loss_grad_direc)
                tower_normal_losses.append(loss_normal)



                regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
                loss = loss_depth + \
                       args.loss_gradient_magnitude_weight*loss_grad_magni + \
                       args.loss_gradient_direction_weight *loss_grad_direc + \
                       args.loss_normal_weight *loss_normal + \
                       regularization_loss
                tower_total_losses.append(loss)

                reuse_vars = True

                with tf.variable_scope("optimizer_vars"):
                    grads = optimizer.compute_gradients(loss)
                tower_grads.append(grads)

        pd_depth_maps_ts = tf.concat(tower_pd_depth_maps, axis=0)


        depth_loss = tf.reduce_mean(tower_depth_losses)
        grad_magni_loss = tf.reduce_mean(tower_grad_magni_losses)
        grad_direc_loss = tf.reduce_mean(tower_grad_direc_losses)
        normal_loss = tf.reduce_mean(tower_normal_losses)

        mean_grads = average_gradients(tower_grads)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.variable_scope("optimizer_vars"):
                train_step = optimizer.apply_gradients(mean_grads, global_step=global_step)


        # make summary
        tf.summary.scalar('depth_loss', depth_loss)
        tf.summary.scalar('grad_magni_loss', grad_magni_loss)
        tf.summary.scalar('grad_direc_loss', grad_direc_loss)
        tf.summary.scalar('normal_loss', normal_loss)

        tf.summary.image('image', batch_images_ts)
        tf.summary.image("gt_depth", batch_depth_ts)
        tf.summary.image("pd_depth", pd_depth_maps_ts)


        # COUNT PARAMS
        total_num_parameters = 0
        total_num_trainable_parameters = 0
        for variable in tf.global_variables():
            if variable in tf.trainable_variables():
                total_num_trainable_parameters += np.array(variable.get_shape().as_list()).prod()
                total_num_parameters += np.array(variable.get_shape().as_list()).prod()
            elif variable in tf.global_variables(scope=args.cnn_model) \
                    or variable in tf.global_variables(scope="MDR_Net"):
                total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("number of trainable parameters: {}".format(total_num_trainable_parameters))
        print("number of parameters: {}".format(total_num_parameters))

        # Put all summary ops into one op. Produces string when you run it.
        merged_summary_op = tf.summary.merge_all()

        if args.process_id_for_refining:
            variables_to_restore = slim.get_variables_to_restore(
                exclude=["optimizer_vars"])
        elif "resnet" in args.cnn_model:
            variables_to_restore = slim.get_variables_to_restore(
                exclude=[args.cnn_model + "/logits", "optimizer_vars", "MDR_Net"])
        elif "vgg" in args.cnn_model:
            variables_to_restore = slim.get_variables_to_restore(
                exclude=[args.cnn_model + "/fc6", args.cnn_model + "/fc7", args.cnn_model + "/fc8",
                         args.cnn_model + "/conv1/conv1_1/BatchNorm",
                         args.cnn_model + "/conv1/conv1_2/BatchNorm",
                         args.cnn_model + "/conv2/conv2_1/BatchNorm",
                         args.cnn_model + "/conv2/conv2_2/BatchNorm",
                         args.cnn_model + "/conv3/conv3_1/BatchNorm",
                         args.cnn_model + "/conv3/conv3_2/BatchNorm",
                         args.cnn_model + "/conv3/conv3_3/BatchNorm",
                         args.cnn_model + "/conv4/conv4_1/BatchNorm",
                         args.cnn_model + "/conv4/conv4_2/BatchNorm",
                         args.cnn_model + "/conv4/conv4_3/BatchNorm",
                         args.cnn_model + "/conv5/conv5_1/BatchNorm",
                         args.cnn_model + "/conv5/conv5_2/BatchNorm",
                         args.cnn_model + "/conv5/conv5_3/BatchNorm",
                         "optimizer_vars", "MDR_Net"])
        else:
            raise NameError("Process id of a trained model must be provided or cnn_model_name must contain resnet or vgg!")


        # Add ops to restore all the variables.
        restorer = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Create the summary writer -- to write all the tboard_log
            # into a specified file. This file can be later read by tensorboard.
            train_writer = tf.summary.FileWriter(CKPT_PATH + "/train", sess.graph)
            validation_writer = tf.summary.FileWriter(CKPT_PATH + "/validation")
            # Create a saver.
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())

            if args.process_id_for_refining:
                ckpt = tf.train.get_checkpoint_state(CKPT_PATH + '/')
                restorer.restore(sess, ckpt.model_checkpoint_path)
                print("Model checkpoits for process id " + process_str_id + " restored!")
            else:
                # load resnet checkpoints
                try:
                    restorer.restore(sess, args.pretrained_ckpts_path + args.cnn_model + '/' + args.cnn_model + ".ckpt")
                    print("Model checkpoits for " + args.cnn_model + " restored!")
                except FileNotFoundError:
                    print("CNN checkpoints not found. Please download " + args.resnet_model +
                          " model checkpoints from: https://github.com/tensorflow/models/tree/master/research/slim")

            # The `Iterator.string_handle()` method returns a tensor that can be evaluated
            # and used to feed the `handle` placeholder.
            training_handle = sess.run(training_iterator.string_handle())
            validation_handle = sess.run(validation_iterator.string_handle())

            sess.run(training_iterator.initializer)


            train_steps_before_eval = 1000
            validation_steps = 10
            saving_steps = 5000
            start_step = global_step.eval(session=sess)
            start_time = time.time()
            before_op_time = start_time
            training_average_loss = 0
            for step in range(start_step, num_total_steps):
                _, global_step_np, train_loss, summary_string = sess.run([train_step,
                                                                          global_step,
                                                                          depth_loss,
                                                                          merged_summary_op],
                                                                          feed_dict={is_training_ts: True,
                                                                                     handle: training_handle})

                if step % 10 == 0:
                    train_writer.add_summary(summary_string, global_step_np)
                training_average_loss += train_loss


                if (step+1)%train_steps_before_eval==0:
                    training_average_loss /= train_steps_before_eval
                    duration = time.time() - before_op_time
                    examples_per_sec = train_steps_before_eval*args.batch_size*args.num_gpus / duration
                    time_sofar = (time.time() - start_time) / 3600
                    training_time_left = (num_total_steps / step - 1.0) * time_sofar

                    print_string = 'Batch {:>6d}| training loss: {:8.5f} |examples/s: {:4.2f} | time elapsed: {:.2f}h |' \
                                   ' time left: {:.2f}h'
                    print(print_string.format(step+1,
                                              training_average_loss,
                                              examples_per_sec,
                                              time_sofar,
                                              training_time_left))
                    training_average_loss = 0

                    sess.run(validation_iterator.initializer)
                    validation_average_loss = 0
                    abs_rel_average = 0
                    sq_rel_average = 0
                    rmse_average = 0
                    rmse_log_average = 0
                    log10_average = 0
                    acc1_average = 0
                    acc2_average = 0
                    acc3_average = 0

                    for i in range(validation_steps):
                        pd_depth_maps, depths, val_loss, summary_string = sess.run([pd_depth_maps_ts,
                                                                                    batch_depth_ts,
                                                                                    depth_loss,
                                                                                    merged_summary_op],
                                                                                    feed_dict={is_training_ts: False,
                                                                                           handle: validation_handle})
                        validation_average_loss += val_loss

                        ABS_REL, SQ_REL, RMSE, RMSE_log, Log10, ACC1, ACC2, ACC3 = \
                            compute_metrics_for_multi_maps(depths, pd_depth_maps)
                        abs_rel_average += ABS_REL
                        sq_rel_average += SQ_REL
                        rmse_average += RMSE
                        rmse_log_average += RMSE_log
                        log10_average += Log10
                        acc1_average += ACC1
                        acc2_average += ACC2
                        acc3_average += ACC3

                    validation_average_loss /= validation_steps
                    abs_rel_average /= validation_steps
                    sq_rel_average /= validation_steps
                    rmse_average /= validation_steps
                    rmse_log_average /= validation_steps
                    log10_average /= validation_steps
                    acc1_average /= validation_steps
                    acc2_average /= validation_steps
                    acc3_average /= validation_steps
                    print_string = 'Batch {:>6d}| validation loss: {:5.3f} | abs_rel: {:2.3f} | sq_rel: {:2.3f} '\
                                   '| rmse: {:2.3f} | rmse_log: {:2.3f} |log10: {:2.3f} ' \
                                    '| acc1: {:.3f} | acc2: {:.3f} |acc3: {:.3f}'
                    print(print_string.format(step+1,
                                              validation_average_loss,
                                              abs_rel_average,
                                              sq_rel_average,
                                              rmse_average,
                                              rmse_log_average,
                                              log10_average,
                                              acc1_average,
                                              acc2_average,
                                              acc3_average))
                    validation_writer.add_summary(summary_string, global_step_np)
                    before_op_time = time.time()
                if step and step % saving_steps == 0:
                    saver.save(sess, CKPT_PATH +"/model.ckpt", global_step=step)

            saver.save(sess, CKPT_PATH +"/model.ckpt", global_step=num_total_steps)
        train_writer.close()

if __name__ == '__main__':
  tf.app.run()
