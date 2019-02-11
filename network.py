import tensorflow as tf
slim = tf.contrib.slim
from nets import resnet_v2, resnet_utils, vgg

# ImageNet mean statistics
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def mdr_arg_scope(weight_decay=0.0001,
                  is_training=True,
                  batch_norm_decay=0.997,
                  batch_norm_epsilon=1e-5,
                  batch_norm_scale=True,
                  activation_fn=tf.nn.relu,
                  use_batch_norm=True):

  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      'is_training': is_training,
      'fused': True,  # Use fused batch norm if possible.
  }

  with slim.arg_scope(
      [slim.conv2d, slim.separable_conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=activation_fn,
      normalizer_fn=slim.batch_norm if use_batch_norm else None,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc


@slim.add_arg_scope
def atrous_spatial_pyramid_pooling(net, scope, is_training, fea_map_size, aspp_rates=[6,12,18], depth=256, reuse=None):
    """
    ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
    (all with 256 filters and batch normalization), and (b) the image-level features as described in https://arxiv.org/abs/1706.05587
    :param net: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
    :param scope: scope name of the aspp layer
    :return: network layer with aspp applyed to it.
    """
    with tf.variable_scope(scope, reuse=reuse):


        # apply global average pooling
        image_level_features = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keepdims=True)
        image_level_features = slim.conv2d(image_level_features, depth, [1, 1], scope="image_level_conv_1x1")
        image_level_features = tf.image.resize_bilinear(image_level_features,
                                                               (fea_map_size[0], fea_map_size[1]),
                                                               align_corners=True)


        at_pool1x1 = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_0_0")
        at_pool3x3_1 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_1_0", rate=aspp_rates[0])
        at_pool3x3_2 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_2_0", rate=aspp_rates[1])
        at_pool3x3_3 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_3_0", rate=aspp_rates[2])

        net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3),
                        axis=3,
                        name="concat")
        net = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_output")

        return net


def mdr_net(inputs, args, image_size, is_training, reuse):

    # mean subtraction normalization
    inputs = inputs - [_R_MEAN, _G_MEAN, _B_MEAN]

    if "resnet" in args.cnn_model:
        # inputs has shape - Original: [batch, height, width, 3]
        with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regularizer,
                                                          is_training,
                                                          args.batch_norm_decay,
                                                          args.batch_norm_epsilon)):
            resnet = getattr(resnet_v2, args.cnn_model)
            net, end_points = resnet(inputs,
                                     multi_grid=args.multi_grid,
                                     output_stride=args.output_stride,
                                     global_pool=False,
                                     num_classes=None,
                                     reuse=reuse)
        lower_level_features = end_points[args.cnn_model + '/block1/unit_3/bottleneck_v2/conv1']
        # low_level_features = end_points[args.cnn_model + 'block1/unit_2/bottleneck_v1/conv3']
    elif "vgg" in args.cnn_model:
        with slim.arg_scope(vgg.vgg_arg_scope(args.l2_regularizer,
                                              is_training,
                                              args.batch_norm_decay,
                                              args.batch_norm_epsilon)):
            net, end_points = vgg.vgg_16(inputs,
                                         multi_grid=args.multi_grid,
                                         output_stride=args.output_stride,
                                         reuse=reuse)
        lower_level_features = end_points[args.cnn_model + '/pool2']
    else:
        raise NameError("cnn_model must contain resnet or vgg!")

    feature_map_size = [int((sz-1)/args.output_stride+1) for sz in image_size]

    arg_sc = mdr_arg_scope(args.l2_regularizer,
                           is_training,
                           args.batch_norm_decay,
                           args.batch_norm_epsilon)

    with slim.arg_scope(arg_sc):
        with tf.variable_scope("MDR_Net", reuse=reuse):

            encoder_output = atrous_spatial_pyramid_pooling(net,
                                                            "ASPP_layer",
                                                            is_training,
                                                            feature_map_size,
                                                            args.aspp_rates,
                                                            depth=256,
                                                            reuse=reuse)

            with tf.variable_scope("decoder", reuse=reuse):
                decoder_depth_1 = 256
                if args.decoding_at_image_size:
                    decoder_depth_2 = 16
                else:
                    decoder_depth_2 = 1
                lower_level_feature_depth = 48
                with tf.variable_scope("lower_level_features"):
                    lower_level_features = slim.conv2d(lower_level_features,
                                                       lower_level_feature_depth,
                                                       [1, 1],
                                                       stride=1,
                                                       scope='conv_1x1')
                    lower_level_features_size = tf.shape(lower_level_features)[1:3]
                with tf.variable_scope("upsampling_logits_1"):
                    net = tf.image.resize_bilinear(encoder_output,
                                                   lower_level_features_size,
                                                   name='upsample_1',
                                                   align_corners=True)
                    net = tf.concat([net, lower_level_features], axis=3, name='concat')

                    num_convs = 2
                    decoder_features = slim.repeat(net,
                                                   num_convs,
                                                   slim.conv2d,
                                                   decoder_depth_1,
                                                   3,
                                                   scope='decoder_conv1')


                with tf.variable_scope("upsampling_logits_2"):
                    if args.decoding_at_image_size:
                        decoder_features = slim.conv2d(decoder_features,
                                                       decoder_depth_2,
                                                       [3, 3],
                                                       scope='decoder_conv2')
                        net = tf.image.resize_bilinear(decoder_features,
                                                       image_size,
                                                       name='upsample_2',
                                                       align_corners=True)
                        logits = slim.conv2d(net,
                                             1,
                                             [1, 1],
                                             activation_fn=None,
                                             normalizer_fn=None,
                                             scope='conv_logits')
                    else:
                        decoder_features = slim.conv2d(decoder_features,
                                                       decoder_depth_2,
                                                       [1, 1],
                                                       activation_fn=None,
                                                       normalizer_fn=None,
                                                       scope='conv_logits')
                        logits = tf.image.resize_bilinear(decoder_features,
                                                          image_size,
                                                          name='upsample_2',
                                                          align_corners=True)
    return logits

