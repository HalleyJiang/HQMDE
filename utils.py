from __future__ import absolute_import, division, print_function
import urllib
import tarfile
import os
import tensorflow as tf

# credit: tensorflow

def download_cnn_checkpoint_if_necessary(checkpoints_path, model_name):
    """
    Check if the cnn checkpoints are already downloaded, if not download it
    :param cnn_checkpoints_path: string: path where the properly cnn checkpoint files should be found
    :param model_name: one of resnet_v2_50, resnet_v2_101, resnet_v2_152, vgg_16 or vgg_19
    :return: None
    """
    resnet_checkpoints_path = checkpoints_path+model_name+'/'
    if not os.path.exists(resnet_checkpoints_path):
        # create the path and download the resnet checkpoints
        os.mkdir(resnet_checkpoints_path)
        if "resnet" in model_name:
            filename = model_name + "_2017_04_14.tar.gz"
        elif "vgg" in model_name:
            filename = model_name + "_2016_08_28.tar.gz"
        else:
            raise NameError("model_name must contain resnet or vgg!")

        url = "http://download.tensorflow.org/models/" + filename
        full_file_path = os.path.join(resnet_checkpoints_path, filename)
        urllib.request.urlretrieve(url, full_file_path)
        thetarfile = tarfile.open(full_file_path, "r:gz")
        thetarfile.extractall(path=resnet_checkpoints_path)
        thetarfile.close()
        print("CNN:", model_name, "successfully downloaded.")
    else:
        print("CNN checkpoints file successfully found.")



# By default, all variables will be placed on '/gpu:0'
# So we need a custom device function, to assign all variables to '/cpu:0'
# Note: If GPUs are peered, '/gpu:0' can be a faster option
PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']

def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign


# Build the function to average the gradients
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads