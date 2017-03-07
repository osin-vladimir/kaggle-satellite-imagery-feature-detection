# required libs
from __future__ import division
import cv2
import sys
import csv
csv.field_size_limit(sys.maxsize)
import shapely.wkt
import shapely.affinity
import numpy as np
import tifffile as tiff

from collections import defaultdict
from shapely.geometry import MultiPolygon, Polygon


class Helper:

    def __init__(self):
        self.data = []

        self.train_polygons = None
        self.pred_polygons = None
        self.train_polygons_scaled = None
        self.pred_polygons_scaled = None

        self.x_scaler = 0
        self.y_scaler = 0

    def load_im_polymask(self, img_id, polygone_type, image_fname, wkt_fname, grid_sizes_fname):
        """"loading for RGB image and mask of it polygones"""

        # Read image with tiff
        im_rgb = tiff.imread(image_fname + '{}.tif'.format(img_id)).transpose([1, 2, 0])
        im_size = im_rgb.shape[:2]

        # get grid sizes for particular image
        x_max, y_min = self.get_grid_sizes(img_id, grid_sizes_fname)

        # load polygon for particular image
        for _im_id, _poly_type, _poly in csv.reader(open(wkt_fname)):
            if _im_id == img_id and _poly_type == polygone_type:
                self.train_polygons = shapely.wkt.loads(_poly)
                break

        # get scales for polygons
        self.scale_polygons(x_max, y_min, im_size)

        # get mask for scaled polygons
        mask_poly = self.mask_for_polygons(self.train_polygons_scaled, im_size)

        return im_rgb, mask_poly

    def mask_for_polygons(self, polygons, im_size):
        """create mask for wkt polygone format"""

        img_mask = np.zeros(im_size)
        if not polygons:
            return img_mask

        int_coords = lambda x: np.array(x).round().astype(np.int32)
        exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
        interiors = [int_coords(pi.coords) for poly in polygons for pi in poly.interiors]
        cv2.fillPoly(img_mask, exteriors, 1)
        cv2.fillPoly(img_mask, interiors, 0)

        return img_mask

    def get_grid_sizes(self, im_id, grid_sizes_fname):
        """find grid sizes for particular image"""
        x_max = y_min = None

        for _im_id, _x, _y in csv.reader(open(grid_sizes_fname)):
            if _im_id == im_id:
                x_max, y_min = float(_x), float(_y)
                break

        return x_max, y_min

    def get_scalers(self, x_max, y_min, im_size):
        """scalers for image, you will need them later for predictions"""

        h, w = im_size  # they are flipped so that mask_for_polygons works correctly
        w_ = w * (w / (w + 1))
        h_ = h * (h / (h + 1))
        return w_ / x_max, h_ / y_min

    def scale_polygons(self, x_max, y_min, im_size, scale_in=True):

        self.x_scaler, self.y_scaler = self.get_scalers(x_max, y_min, im_size)

        if scale_in:
            self.train_polygons_scaled = shapely.affinity.scale(self.train_polygons,
                                                           xfact=self.x_scaler,
                                                           yfact=self.y_scaler,
                                                           origin=(0, 0, 0))
        else:
            self.pred_polygons_scaled = shapely.affinity.scale(self.pred_polygons,
                                                          xfact=1 / self.x_scaler,
                                                          yfact=1 / self.y_scaler,
                                                          origin=(0, 0, 0))

    def mask_to_polygons(self, mask, epsilon=0.01, min_area=2., max_area=100, buffer_value=0.0001):
        """transform predicted mask to wkt format"""

        # first, find contours with cv2: it's much faster than shapely
        image, contours, hierarchy = cv2.findContours(
            ((mask == 1) * 255).astype(np.uint8),
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_TC89_KCOS)

        # create approximate contours to have reasonable submission size
        approx_contours = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]

        if not contours:
            return MultiPolygon()

        # now messy stuff to associate parent and child contours
        cnt_children = defaultdict(list)
        child_contours = set()
        assert hierarchy.shape[0] == 1

        # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
        for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
            if parent_idx != -1:
                child_contours.add(idx)
                cnt_children[parent_idx].append(approx_contours[idx])

        # create actual polygons filtering by area (removes artifacts)
        all_polygons = []
        for idx, cnt in enumerate(approx_contours):
            if idx not in child_contours and cv2.contourArea(cnt) >= min_area and cv2.contourArea(cnt)<max_area:
                assert cnt.shape[1] == 1
                poly = Polygon(
                    shell=cnt[:, 0, :],
                    holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                           if cv2.contourArea(c) >= min_area and cv2.contourArea(c) <= max_area])
                all_polygons.append(poly)

        # approximating polygons might have created invalid ones, fix them
        all_polygons = MultiPolygon(all_polygons)
        if not all_polygons.is_valid:
            all_polygons = all_polygons.buffer(buffer_value)

            # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
            # need to keep it a Multi throughout
            if all_polygons.type == 'Polygon':
                all_polygons = MultiPolygon([all_polygons])
        return all_polygons

    def show_mask(self, m):
        tiff.imshow(np.stack([m, m, m]))

    def show_img(self, img_rgb):
        tiff.imshow(self.scale_percentile(img_rgb))

    def load_img(self, fname):
        im_rgb = tiff.imread(fname)
        return im_rgb

    def load_tiff_img(self, fname, img_id):
        im_rgb = tiff.imread(fname + '{}.tif'.format(img_id)).transpose([1, 2, 0])
        im_size = im_rgb.shape[:2]
        return im_rgb, im_size

    def scale_percentile(self, matrix):
        w, h, d = matrix.shape
        matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)

        # Get 2nd and 98th percentile
        mins = np.percentile(matrix, 1, axis=0)
        maxs = np.percentile(matrix, 99, axis=0) - mins
        matrix = (matrix - mins[None, :]) / maxs[None, :]
        matrix = np.reshape(matrix, [w, h, d])
        matrix = matrix.clip(0, 1)

        return matrix


