import os
import sys

import numpy as np
import tensorflow as tf
import random

from ..core.input import read_png_image, Input
from ..core.augment import random_crop
#from ..core.flow_read_write import readPFM,writePFM


def _read_flow(filenames, num_epochs=None):   #
    """Given a list of filenames, constructs a reader op for ground truth flow files."""
    filename_queue = tf.train.string_input_producer(filenames,
        shuffle=False, capacity=len(filenames), num_epochs=num_epochs)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    value = tf.reshape(value, [1])
    value_width = tf.substr(value, 4, 4)
    value_height = tf.substr(value, 8, 4)
    width = tf.reshape(tf.decode_raw(value_width, out_type=tf.int32), [])
    height = tf.reshape(tf.decode_raw(value_height, out_type=tf.int32), [])

    value_flow = tf.substr(value, 12, 8 * width * height)
    flow = tf.decode_raw(value_flow, out_type=tf.float32)
    flow = tf.reshape(flow,[height, width, 2])
    mask = tf.to_float(tf.logical_and(flow[:, :, 0] < 1e9, flow[:, :, 1] < 1e9))
    mask = tf.reshape(mask, [height, width, 1])

    return flow, mask
    # return tf.reshape(flow, [436, 1024, 2])


def _read_binary(filenames, num_epochs=None):
    """Given a list of filenames, constructs a reader op for ground truth binary files."""
    filename_queue = tf.train.string_input_producer(filenames,
        shuffle=False, capacity=len(filenames), num_epochs=num_epochs)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    value_decoded = tf.image.decode_png(value, channels=1)
    return tf.cast(value_decoded, tf.float32)


def _get_filenames(parent_dir, ignore_last=False):
    filenames = []
    for sub_name in sorted(os.listdir(parent_dir)):
        sub_dir = os.path.join(parent_dir, sub_name)
        sub_filenames = os.listdir(sub_dir)
        sub_filenames.sort()
        if ignore_last:
            sub_filenames = sub_filenames[:-1]     #到最后一个元素之前的所有元素
        for filename in sub_filenames:
            filenames.append(os.path.join(sub_dir, filename))

    return filenames


class MonkaaInput(Input):
    def __init__(self, data, batch_size, dims, *,
                 num_threads=1, normalize=True):
        super().__init__(data, batch_size, dims, num_threads=num_threads,
                         normalize=normalize)

    def _preprocess_flow(self, t, channels):
        height, width = self.dims
        # Reshape to tell tensorflow we know the size statically
        return tf.reshape(self._resize_crop_or_pad(t), [height, width, channels])

    def _input_images(self, image_dir):
        """Assumes that paired images are next to each other after ordering the
        files.
        """
        image_dir = os.path.join(self.data.current_dir, image_dir)

        filenames_1 = []
        filenames_2 = []

        for sub_name in sorted(os.listdir(image_dir)):
            sub_dir = os.path.join(image_dir, sub_name)
            sub_filenames = os.listdir(sub_dir)
            sub_filenames.sort()
            for i in range(len(sub_filenames) - 1):
                filenames_1.append(os.path.join(sub_dir, sub_filenames[i]))
                filenames_2.append(os.path.join(sub_dir, sub_filenames[i + 1]))

        input_1 = read_png_image(filenames_1, 1)
        input_2 = read_png_image(filenames_2, 1)
        image_1 = self._preprocess_image(input_1)
        image_2 = self._preprocess_image(input_2)
        return tf.shape(input_1), image_1, image_2

    def _input_flow(self):
        flow_dir = os.path.join(self.data.current_dir, 'monkaa_frames_finalpass/flow')# 这里要flow
        # invalid_dir = os.path.join(self.data.current_dir, 'sintel/training/invalid')
        # occ_dir = os.path.join(self.data.current_dir, 'sintel/training/occlusions')
        flow_files = _get_filenames(flow_dir)
        # invalid_files = _get_filenames(invalid_dir, ignore_last=True)
        # occ_files = _get_filenames(occ_dir)

        # assert len(flow_files) == len(invalid_files) == len(occ_files)

        flow = self._preprocess_flow(_read_flow(flow_files, 1), 2)
        mask = self._preprocess_flow(_read_flow(flow_files, 1), 1)
        # invalid = self._preprocess_flow(_read_binary(invalid_files), 1)
        # occ = self._preprocess_flow(_read_binary(occ_files), 1)

        # flow_occ = flow  # occluded
        # flow_noc = flow * (1 - occ)  # non-occluded
        # mask_occ = (1 - invalid)
        # mask_noc = mask_occ * (1 - occ)

        return flow, mask

    def _input_train(self, image_dir):
        input_shape, im1, im2 = self._input_images(image_dir)
        flow, mask = self._input_flow()
        return tf.train.batch(
            [im1, im2, input_shape, flow, mask],
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            allow_smaller_final_batch=True)
    ###
    def input_train_gt(self):
        img_dirs = ['data_sf/monkaa_frames_finalpass/training',
                    'data_sf/monkaa_frames_finalpass/fine_tune_training',
                    'data_sf/monkaa_frames_finalpass/test',
                    'data_sf/monkaa_frames_finalpass/fine_tune_test']
        gt_dirs = ['pengfei/LI XINYAO/monkaa_frames_finalpass_flo/training_flow',
                   'pengfei/LI XINYAO/monkaa_frames_finalpass_flo/fine_tune_training_flow',
                   'pengfei/LI XINYAO/monkaa_frames_finalpass_flo/test_flow',
                   'pengfei/LI XINYAO/monkaa_frames_finalpass_flo/fine_tune_test_flow']

        height, width = self.dims

        filenames = []
        for img_dir, gt_dir in zip(img_dirs, gt_dirs):#zip()是Python的一个内建函数，它接受一系列可迭代的对象作为参数，将对象中对应的元素打包成一个个tuple（元组），然后返回由这些tuples组成的list（列表）。
            dataset_filenames = []
            img_dir = os.path.join(self.data.current_dir, img_dir)
            gt_dir = os.path.join(self.data.current_dir, gt_dir)
            img_files = os.listdir(img_dir)
            gt_files = os.listdir(gt_dir)
            img_files.sort()#sort排序，分类
            gt_files.sort()
            assert len(img_files) % 2 == 0 and len(img_files) / 2 == len(gt_files)

            for i in range(len(gt_files)):
                # fn_im1 = os.path.join(img_dir, img_files[2 * i])#even
                # fn_im2 = os.path.join(img_dir, img_files[2 * i + 1])#odd
                fn_im = os.path.join(img_dir, img_files[i])
                fn_gt = os.path.join(gt_dir, gt_files[i])
                dataset_filenames.append((fn_im, fn_gt))

            # random.seed(0)
            # random.shuffle(dataset_filenames)
            # dataset_filenames = dataset_filenames[hold_out:]
            filenames.extend(dataset_filenames)

        # random.seed(0)
        # random.shuffle(filenames)

        #shift = shift % len(filenames)
        #filenames_ = list(np.roll(filenames, shift))

        # fns_im1, fns_im2, fns_gt = zip(*filenames)
        fns_im, fns_gt = zip(*filenames)
        # fns_im1 = list(fns_im1)
        # fns_im2 = list(fns_im2)
        fns_im = list(fns_im)
        fns_gt = list(fns_gt)

        # im1 = read_png_image(fns_im1)
        # im2 = read_png_image(fns_im2)
        im = read_png_image(fns_im)
        flow_gt, mask_gt = _read_flow(fns_gt)

        gt_queue = tf.train.string_input_producer(fns_gt,
            shuffle=False, capacity=len(fns_gt), num_epochs=None)
        reader = tf.WholeFileReader()
        _, gt_value = reader.read(gt_queue)
        gt_uint16 = tf.image.decode_png(gt_value, dtype=tf.uint16) #tensorflow里面给出了一个函数用来读取图像，不过得到的结果是最原始的图像，是没有经过解码的图像，这个函数为tf.gfile.FastGFile（‘path’， ‘r’）.read()。如果要显示读入的图像，那就需要经过解码过程，tensorflow里面提供解码的函数有两个，tf.image.decode_jepg和tf.image.decode_png分别用于解码jpg格式和png格式的图像进行解码，得到图像的像素值，这个像素值可以用于显示图像。如果没有解码，读取的图像是一个字符串，没法显示。
        gt = tf.cast(gt_uint16, tf.float32) # 类型转换

        # im1, im2, gt = random_crop([im1, im2, gt],
        #                            [height, width, 3])
        im, gt = random_crop([im, gt],
                             [height, width, 3])
        flow_gt = (gt[:, :, 0:2] - 2 ** 15) / 64.0
        mask_gt = gt[:, :, 2:3]

        if self.normalize:
            im = self._normalize_image(im)
            # im1 = self._normalize_image(im1)
            # im2 = self._normalize_image(im2)

        return tf.train.batch(
            [im, flow_gt, mask_gt],
            batch_size=self.batch_size,
            num_threads=self.num_threads)

        # return tf.train.batch(
        #     [im1, im2, flow_gt, mask_gt],
        #     batch_size=self.batch_size,
        #     num_threads=self.num_threads)

    def input_train_unsuper(self):
        return self._input_train('monkaa_frames_finalpass/training')

    def input_train_ft(self):
        return self._input_train('monkaa_frames_finalpass/fine_tune_training')

    def input_test_unsuper(self):
        input_shape, im1, im2 = self._input_images('monkaa_frames_finalpass/test')
        return tf.train.batch(
           [im1, im2, input_shape],
           batch_size=self.batch_size,
           num_threads=self.num_threads,
           allow_smaller_final_batch=True)

    def input_test_ft(self):
        input_shape, im1, im2 = self._input_images('monkaa_frames_finalpass/fine_tune_test')
        return tf.train.batch(
           [im1, im2, input_shape],
           batch_size=self.batch_size,
           num_threads=self.num_threads,
           allow_smaller_final_batch=True)