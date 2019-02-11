import tensorflow as tf


def berhu(targets, predictions):
    residual = tf.abs(predictions - targets)
    max_residual = tf.reduce_max(residual, axis=0)
    delta = 0.2 * max_residual
    condition = tf.less(residual, delta)
    large_res = 0.5 * (delta + tf.square(residual) / delta)
    return tf.where(condition, residual, large_res)


def compute_depth_loss(gt_depth_maps, pred_depth_maps, loss_depth_norm, using_log_depth):

    if using_log_depth:
        gt_depth_maps = tf.log(gt_depth_maps+0.5)
        pred_depth_maps = tf.log(pred_depth_maps+0.5)
    if loss_depth_norm == 'l1':
        depth_diff = tf.abs(gt_depth_maps - pred_depth_maps)
    elif loss_depth_norm == 'l2':
        depth_diff = tf.square(gt_depth_maps - pred_depth_maps)
    elif loss_depth_norm == 'berhu':
        depth_diff = berhu(gt_depth_maps, pred_depth_maps)
    else:
        raise NameError("loss_depth_norm must l1, l2 or beuhu.")
    loss_depth = tf.reduce_mean(depth_diff)

    return loss_depth


def compute_gradient_loss(gt_depth_maps, pred_depth_maps, using_log_gradient_magnitude, loss_gradient_magnitude_norm, loss_gradient_direction_norm):

    gt_gradients = tf.image.sobel_edges(gt_depth_maps)
    pd_gradients = tf.image.sobel_edges(pred_depth_maps)
    gt_gradients_y = gt_gradients[:, :, :, :, 0]
    gt_gradients_x = gt_gradients[:, :, :, :, 1]
    pd_gradients_y = pd_gradients[:, :, :, :, 0]
    pd_gradients_x = pd_gradients[:, :, :, :, 1]

    if loss_gradient_direction_norm == 'l1':
        grad_direc_diff = tf.abs(gt_gradients_y*pd_gradients_x - gt_gradients_x*pd_gradients_y)
    elif loss_gradient_direction_norm == 'l2':
        grad_direc_diff = tf.square(gt_gradients_y*pd_gradients_x - gt_gradients_x*pd_gradients_y)
    elif loss_gradient_direction_norm == 'berhu':
        grad_direc_diff = berhu(gt_gradients_y*pd_gradients_x, gt_gradients_x*pd_gradients_y)
    else:
        raise NameError("loss_gradient_magnitude_norm must l1, l2 or beuhu.")
    loss_grad_direc = tf.reduce_mean(grad_direc_diff)

    normal_product = 1 + gt_gradients_y*pd_gradients_y + gt_gradients_x*pd_gradients_x
    gt_normal_magni = tf.sqrt(1 + tf.square(gt_gradients_y) + tf.square(gt_gradients_x))
    pd_normal_magni = tf.sqrt(1 + tf.square(pd_gradients_y) + tf.square(pd_gradients_x))
    loss_normal = tf.reduce_mean(1 - tf.divide(normal_product, gt_normal_magni*pd_normal_magni))

    if using_log_gradient_magnitude:
        gt_gradients = tf.log(gt_gradients_y+0.5)
        pd_gradients = tf.log(pd_gradients_y+0.5)

    if loss_gradient_magnitude_norm == 'l1':
        grad_magni_diff = tf.abs(gt_gradients - pd_gradients)
    elif loss_gradient_magnitude_norm == 'l2':
        grad_magni_diff = tf.square(gt_gradients - pd_gradients)
    elif loss_gradient_magnitude_norm == 'berhu':
        grad_magni_diff = berhu(gt_gradients, pd_gradients)
    else:
        raise NameError("loss_gradient_magnitude_norm must l1, l2 or beuhu.")
    loss_grad_magni = tf.reduce_mean(grad_magni_diff)

    return loss_grad_magni, loss_grad_direc, loss_normal







