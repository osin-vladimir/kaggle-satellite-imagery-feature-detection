from __future__ import division
import mxnet as mx
import numpy as np
from mxnet.io import DataIter
from mxnet.io import DataBatch
import random
import tifffile as tiff
from helpers import Helper
import cv2


class FileIter(DataIter):

    def __init__(self, root_dir, data_list, batch_size, iter_num, cut_off_size=None, data_name="data", label_name="label"):

        super(FileIter, self).__init__()
        self.data_list = data_list
        self.root_dir = root_dir
        self.cut_off_size = cut_off_size
        self.data_name = data_name
        self.label_name = label_name
        self.random = random.Random()
        self.cursor = -1
        self.batch_size = batch_size
        self.iter_num = iter_num
        self.data, self.label = self._read()

    def _read(self):
        data = {}
        label = {}

        dd = []
        ll = []

        for i in range(0, self.batch_size):
            rand = random.Random()
            ind = rand.randint(0, len(self.data_list) - 1)

            d, l = self._read_img(self.data_list[ind])
            dd.append(d)
            ll.append(l)

        d = np.vstack(dd)
        l = np.vstack(ll)

        data[self.data_name] = d
        label[self.label_name] = l

        res = list(data.items()), list(label.items())
        return res

    def _read_img(self, IM_ID):

        helper = Helper()

        # read rgb channels and mask
        rgb, mask = helper.load_im_polymask(IM_ID, '5', 'data/three_band/', 'data/train_wkt_v4.csv', 'data/grid_sizes.csv')
        # read m band
        m = tiff.imread(self.root_dir+'{}_M.tif'.format(IM_ID))

        shape_0 = 3345
        shape_1 = 3338

        # rgb
        blue = cv2.resize(m[1, :, :], (shape_1, shape_0), interpolation=cv2.INTER_CUBIC)[:, :, np.newaxis]
        green = cv2.resize(m[2, :, :], (shape_1, shape_0), interpolation=cv2.INTER_CUBIC)[:, :, np.newaxis]
        red = cv2.resize(m[4, :, :], (shape_1, shape_0), interpolation=cv2.INTER_CUBIC)[:, :, np.newaxis]

        # nir1 = cv2.resize(m[6, :, :], (shape_1, shape_0), interpolation=cv2.INTER_CUBIC)[:, :, np.newaxis]
        # nir2 = cv2.resize(m[7, :, :], (shape_1, shape_0), interpolation=cv2.INTER_CUBIC)[:, :, np.newaxis]

        image = np.concatenate([red, green, blue], axis=2)
        img_rgb_norm = (image - np.array([ 426.98400879, 464.88424683, 329.17596436]))

         # swapaxes
        img_rgb_norm = img_rgb_norm
        img_rgb_norm = np.swapaxes(img_rgb_norm, 0, 2)
        img_rgb_norm = np.swapaxes(img_rgb_norm, 1, 2)
        img_rgb_norm = img_rgb_norm[np.newaxis, :]

        # random crop
        crop_size = 400
        rand = random.Random()

        crop_max_x = img_rgb_norm.shape[3] - crop_size
        crop_max_y = img_rgb_norm.shape[2] - crop_size

        crop_x = rand.randint(0, crop_max_x)
        crop_y = rand.randint(0, crop_max_y)

        img_crop = img_rgb_norm[:, :, crop_y:crop_y + crop_size, crop_x: crop_x + crop_size]
        mask_croped = mask[crop_y:crop_y + crop_size, crop_x: crop_x + crop_size]
        mask_croped = np.expand_dims(mask_croped, axis=0)
        mask_croped = np.expand_dims(mask_croped, axis=0)
        return img_crop, mask_croped

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self.data]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self.label]

    def get_batch_size(self):
        return self.batch_size

    def reset(self):
        self.cursor = -1

    def iter_next(self):
        self.cursor += 1
        if(self.cursor < self.iter_num-1):
            return True
        else:
            return False

    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            self.data, self.label = self._read()
            res = DataBatch(data=[mx.nd.array(self.data[0][1])],
                            label=[mx.nd.array(self.label[0][1])],
                            index=None)
            return res
        else:
            raise StopIteration