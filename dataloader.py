"""deep monocular depth regression data loader. """


from __future__ import absolute_import, division, print_function
import tensorflow as tf

_RESIZE_WIDTH_nyu_depth_v2 = 400
_RESIZE_HEIGHT_nyu_depth_v2 = 300
_INPUT_WIDTH_nyu_depth_v2 = 385
_INPUT_HEIGHT_nyu_depth_v2 = 289
_MAX_DEPTH_nyu_depth_v2 = 10.0
_MIN_DEPTH_nyu_depth_v2 = 0.01


def string_length_tf(t):
  return tf.py_func(len, [t], [tf.int64])

class MDEdataloader(object):
    """MDE dataloader"""

    def __init__(self, dataset='nyu_depth_v2',
                 num_threads=4,
                 batch_size=1,
                 epochs=1,
                 train_file=None,
                 val_file=None,
                 test_file=None):

        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None


        if dataset == 'nyu_depth_v2':
            self.RESIZE_WIDTH = _RESIZE_WIDTH_nyu_depth_v2
            self.RESIZE_HEIGHT = _RESIZE_HEIGHT_nyu_depth_v2
            self.INPUT_WIDTH = _INPUT_WIDTH_nyu_depth_v2
            self.INPUT_HEIGHT = _INPUT_HEIGHT_nyu_depth_v2
            self.MAX_DEPTH = _MAX_DEPTH_nyu_depth_v2
            self.MIN_DEPTH = _MIN_DEPTH_nyu_depth_v2
            self.max_scale = 1.5
            self.min_scale = 1.0
        else:
            raise ValueError('dataset must be nyu_depth_v2.')

        self.num_train_samples = 0
        self.num_val_samples = 0
        self.num_test_samples = 0

        if train_file:
            with open(train_file, 'r') as f:
                train_filelist = [line.strip().split() for line in f.readlines()]
            self.num_train_samples = len(train_filelist)
            self.train_dataset = tf.data.Dataset.from_tensor_slices(train_filelist)

            def train_parse_fn(image_paths):
                rgb, depth = self.read_image(image_paths)
                rgb, depth = self.resize_and_scale(rgb, depth)
                if dataset is 'nyu_depth_v2':
                    rgb, depth = self.rotate(rgb, depth)
                rgb, depth = self.crop(rgb, depth)
                rgb, _ = self.augment_color(rgb)
                rgb, depth = self.random_flip(rgb, depth)
                max_depth = tf.constant(self.MAX_DEPTH, shape=[1, 1, 1])
                return rgb, depth, max_depth

            self.train_dataset = self.train_dataset.map(train_parse_fn, num_parallel_calls=num_threads)
            self.train_dataset = self.train_dataset.repeat()
            self.train_dataset = self.train_dataset.shuffle(buffer_size=500+5*batch_size)
            self.train_dataset = self.train_dataset.batch(batch_size)
            self.train_dataset = self.train_dataset.prefetch(5*batch_size)


        if val_file:
            with open(val_file, 'r') as f:
                val_filelist = [line.strip().split() for line in f.readlines()]
            self.num_val_samples = len(val_filelist)
            """ """
            def val_parse_fn(image_paths):
                rgb, depth = self.read_image(image_paths)
                rgb, depth = self.resize(rgb, depth)
                rgb, depth = self.crop(rgb, depth)
                max_depth = tf.constant(self.MAX_DEPTH, shape=[1, 1, 1])
                return rgb, depth, max_depth


            self.val_dataset = tf.data.Dataset.from_tensor_slices(val_filelist)
            self.val_dataset = self.val_dataset.map(val_parse_fn, num_parallel_calls=num_threads)

            self.val_dataset = self.val_dataset.repeat()
            self.val_dataset = self.val_dataset.shuffle(buffer_size=100+2*batch_size)
            self.val_dataset = self.val_dataset.batch(batch_size)
            self.val_dataset = self.val_dataset.prefetch(2*batch_size)


        if test_file:
            with open(test_file, 'r') as f:
                test_filelist = [line.strip().split() for line in f.readlines()]
            self.num_test_samples = len(test_filelist)
            self.test_dataset = tf.data.Dataset.from_tensor_slices(test_filelist)
            def test_parse_fn(image_paths):
                rgb, depth = self.read_image(image_paths)
                im_size = tf.shape(rgb)[0:2]
                rgb = tf.image.resize_images(rgb, [self.INPUT_HEIGHT, self.INPUT_WIDTH])
                return rgb, depth, im_size

            self.test_dataset = self.test_dataset.map(test_parse_fn, num_parallel_calls=num_threads)
            self.test_dataset = self.test_dataset.batch(1)

    def read_image(self, image_paths):
        rgb_string = tf.read_file(image_paths[0])
        #rgb_decoded = tf.image.decode_image(rgb_string, channels=3)

        path_length = string_length_tf(rgb_string)[0]
        file_extension = tf.substr(rgb_string, path_length-3, 3)
        file_cond = tf.equal(file_extension, 'jpg')

        rgb_decoded = tf.cond(file_cond,
                              lambda: tf.image.decode_jpeg(rgb_string, channels=3),
                              lambda: tf.image.decode_png(rgb_string, channels=3))
        depth_file = tf.read_file(image_paths[1])
        depth_decoded = tf.image.decode_png(depth_file, channels=1, dtype=tf.uint16)

        rgb_float = tf.to_float(rgb_decoded)
        depth_float = tf.divide(tf.to_float(depth_decoded), [256.0])
        return rgb_float, depth_float

    def resize(self, rgb, depth):
        size = tf.shape(rgb)
        downsize_scale = tf.divide(tf.to_float(self.RESIZE_HEIGHT), tf.to_float(size[0])) if self.RESIZE_HEIGHT \
            else tf.divide(tf.to_float(self.RESIZE_WIDTH), tf.to_float(size[1]))
        h_scaled = tf.to_int32(tf.multiply(tf.to_float(size[0]), downsize_scale))
        w_scaled = tf.to_int32(tf.multiply(tf.to_float(size[1]), downsize_scale))
        size_scaled = tf.stack([h_scaled, w_scaled], axis=0)
        rgb_resized = tf.image.resize_images(rgb, size_scaled)
        depth_resized = tf.image.resize_images(depth, size_scaled,
                                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return rgb_resized, depth_resized


    def resize_and_scale(self, rgb, depth):
        size = tf.shape(rgb)
        downsize_scale = tf.divide(tf.to_float(self.RESIZE_HEIGHT), tf.to_float(size[0])) if self.RESIZE_HEIGHT \
            else tf.divide(tf.to_float(self.RESIZE_WIDTH), tf.to_float(size[1]))
        scale = tf.random_uniform([1], minval=self.min_scale, maxval=self.max_scale, dtype=tf.float32, seed=None)
        h_scaled = tf.to_int32(tf.multiply(tf.to_float(size[0]), scale*downsize_scale))
        w_scaled = tf.to_int32(tf.multiply(tf.to_float(size[1]), scale*downsize_scale))
        size_scaled = tf.concat([h_scaled, w_scaled], axis=0)
        rgb_scaled = tf.image.resize_images(rgb, size_scaled)
        depth_scaled = tf.image.resize_images(depth, size_scaled,
                                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        depth_scaled = tf.divide(depth_scaled, scale)
        return rgb_scaled, depth_scaled


    def rotate(self, rgb, depth):
        angle = tf.random_uniform([1], minval=-5, maxval=5, dtype=tf.float32, seed=None)
        angle = angle*3.14/180
        rgb_rotated = tf.contrib.image.rotate(rgb, angle, interpolation='BILINEAR')
        depth_rotated = tf.contrib.image.rotate(depth, angle)
        return rgb_rotated, depth_rotated

    def crop(self, rgb, depth):
        combine = tf.concat(axis=2, values=[rgb, depth])
        combine_cropped = tf.random_crop(combine, [self.INPUT_HEIGHT, self.INPUT_WIDTH, 4])
        rgb_cropped = combine_cropped[:, :, :3]
        depth_cropped = combine_cropped[:, :, 3:]
        rgb_cropped.set_shape((self.INPUT_HEIGHT, self.INPUT_WIDTH, 3))
        depth_cropped.set_shape((self.INPUT_HEIGHT, self.INPUT_WIDTH, 1))
        return rgb_cropped, depth_cropped

    def augment_color(self, rgb, depth=None):
        rgb_normalized = rgb/255

        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        rgb_aug = rgb_normalized ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.8, 1.25)
        rgb_aug = rgb_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(rgb_aug)[0], tf.shape(rgb_aug)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        rgb_aug *= color_image

        # saturate
        rgb_aug = tf.clip_by_value(rgb_aug, 0, 1)
        rgb_aug = rgb_aug*255
        return rgb_aug, depth


    def random_flip(self, rgb, depth):
        random_var = tf.random_uniform([], 0, 1)
        rgb_randomly_flipped = tf.cond(pred=tf.greater(random_var, 0.5),
                                       true_fn=lambda: tf.image.flip_left_right(rgb),
                                       false_fn=lambda: rgb)
        depth_randomly_flipped = tf.cond(pred=tf.greater(random_var, 0.5),
                                         true_fn=lambda: tf.image.flip_left_right(depth),
                                         false_fn=lambda: depth)
        return rgb_randomly_flipped, depth_randomly_flipped
