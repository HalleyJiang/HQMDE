# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

@@vgg_a
@@vgg_16
@@vgg_19
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def vgg_arg_scope(weight_decay=0.0005,
                  is_training=True,
                  batch_norm_decay=0.997,
                  batch_norm_epsilon=1e-5,
                  batch_norm_scale=True,
                  activation_fn=tf.nn.relu,
                  use_batch_norm=True):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  batch_norm_params = {
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
    'updates_collections': tf.GraphKeys.UPDATE_OPS,
    'is_training': is_training,
    'fused': True,  # Use fused batch norm if possible.
  }

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      biases_initializer=tf.zeros_initializer(),
      activation_fn=activation_fn,
      normalizer_fn=slim.batch_norm if use_batch_norm else None,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc


def vgg_16(inputs,
           output_stride=None,
           multi_grid = [1, 2, 4],
           scope='vgg_16',
           reuse=None):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """

  # The current_stride variable keeps track of the effective stride of the
  # activations. This allows us to invoke atrous convolution whenever applying
  # the next residual unit would result in the activations having stride larger
  # than the target output_stride.
  current_stride = 1

  # The atrous convolution rate parameter.
  rate = 1

  with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], rate=rate, scope='conv1', trainable=False)
      if output_stride is None or current_stride < output_stride:
        net = slim.max_pool2d(net, [3, 3], padding='SAME', scope='pool1')
        net = slim.utils.collect_named_outputs(end_points_collection, sc.name + '/pool1', net)
        current_stride *=2
      else:
        rate *=2

      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], rate=rate, scope='conv2', trainable=False)
      if output_stride is None or current_stride < output_stride:
        net = slim.max_pool2d(net, [3, 3], padding='SAME', scope='pool2')
        net = slim.utils.collect_named_outputs(end_points_collection, sc.name + '/pool2', net)
        current_stride *=2
      else:
        rate *=2

      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], rate=rate, scope='conv3')
      if output_stride is None or current_stride < output_stride:
        net = slim.max_pool2d(net, [3, 3], padding='SAME', scope='pool3')
        net = slim.utils.collect_named_outputs(end_points_collection, sc.name + '/pool3', net)
        current_stride *=2
      else:
        rate *=2

      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], rate=rate, scope='conv4')
      if output_stride is None or current_stride < output_stride:
        net = slim.max_pool2d(net, [3, 3], padding='SAME', scope='pool4')
        net = slim.utils.collect_named_outputs(end_points_collection, sc.name+ '/pool4', net)
        current_stride *=2
      else:
        rate *=2
      #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')

      net = slim.conv2d(net, 512, [3, 3], rate=rate*multi_grid[0], scope='conv5/conv5_1')
      net = slim.conv2d(net, 512, [3, 3], rate=rate*multi_grid[1], scope='conv5/conv5_2')
      net = slim.conv2d(net, 512, [3, 3], rate=rate*multi_grid[2], scope='conv5/conv5_3')

      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)

      return net, end_points


vgg_16.default_image_size = 224
vgg_d = vgg_16


