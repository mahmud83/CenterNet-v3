from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import time
import os
import cv2
import numpy as np
from opts import opts
from detectors.detector_factory import detector_factory
import torch
from shapely.geometry import Polygon, MultiPoint
from PIL import Image, ImageDraw, ImageFont
import shapely
import copy
from show_utils import show_bbox, show_bbox_angle, area_and_point_nms
from nms_iou.nms import nms

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

color_list = np.array(
        [
            1.000, 1.000, 1.000,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255

num_classes = 2
# dota_class_name = ['__background__','plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
#                 'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter'
#             ]
dota_class_name = ['__background__', 'small-vehicle', 'large-vehicle']
ZERO = 1e-9
def show_line(img, h_overlaps, w_overlaps):
    count = 0
    for h in h_overlaps:
        for w in w_overlaps:
            count += 1
            h_s = int(h)
            w_s = int(w)
            cv2.rectangle(img, (w_s, h_s), (w_s + 1024, h_s + 1024), (0, 0, 255), 3)
    return img


def bbox_txt_ready_nms(bboxes, result, base_name):
    # file_det = open(txt, 'a')
    for bbox in bboxes:
        # for idx, bbox in enumerate(bboxes[i]):
        cls_name = int(bbox[9])
        score = bbox[8]
        bbox = list(map(int, list(bbox[0:8])))
        s = '{} {:.3f} {} {} {} {} {} {} {} {}\n'.format(base_name, score,
                                                         bbox[0], bbox[1],
                                                         bbox[2], bbox[3],
                                                         bbox[4], bbox[5],
                                                         bbox[6], bbox[7])
        result[dota_class_name[cls_name]] = result.get(dota_class_name[cls_name], []) + [s]
    return result


def judge_bbox_duplicate(bbox1, bbox2):
    inter = []
    for i in range(1, 1 + num_classes):
        bbox1_cls = list(map(list, bbox1[i][:, 5:13]))
        bbox2_cls = list(map(list, bbox2[i][:, 5:13]))
        inter.extend([box1 for box1 in bbox1_cls if box1 in bbox2_cls])
    return inter


def create_txt(txt_path, result):
    if not os.path.exists(txt_path):
        os.mkdir(txt_path)
    for key, value in result.items():
        f = open('{}/Task1_{}.txt'.format(txt_path, key), 'w')
        for v in value:
            f.write(v)
        f.close()

def patchImg(img, resolution=1, scale=512, overlap=0.25):
    height_i, width_i, depth_i = img.shape
    # normal h w is larger than scale
    h, w, d = int(height_i * resolution), int(width_i * resolution), depth_i
    img_resolution = cv2.resize(img, (w, h))
    h_new = int(max([h, scale]))
    w_new = int(max([w, scale]))
    img_resolution_new = np.zeros((h_new, w_new, d), dtype=np.uint8)
    img_resolution_new[0:h, 0:w, :] = img_resolution.copy()
    img_resolution = img_resolution_new

    imgs = []
    h_min, h_max = 0, int(scale)
    h_overlaps = []
    w_overlaps = []
    while h_max <= h_new:
        w_min, w_max = 0, int(scale)
        w_overlaps = []

        while w_max <= w_new:
            img = img_resolution[h_min:h_max, w_min:w_max, :]
            imgs.append(img)
            w_overlaps.append(w_min / scale)
            # update
            if w_max == w_new:
                break
            w_max = int(min([w_max + scale * (1 - overlap), w_new]))
            w_min = int(w_max - scale)

        h_overlaps.append(h_min / scale)
        # update
        if h_max == h_new:
            break
        h_max = int(min([h_max + scale * (1 - overlap), h_new]))
        h_min = int(h_max - scale)
    # img_resolution origin img
    return imgs, h_overlaps, w_overlaps, img_resolution


def uniform_bbox_nms(h_overlaps, w_overlaps, key_name, results, vis_thresh):
    # boxes = []
    bbox_total = {}
    bbox_four = []
    bbox_right = {}
    bbox_right_list = []
    bbox_bottom = {}

    h_len = len(h_overlaps)
    for i in range(len(h_overlaps)):
        w_len = len(w_overlaps)
        for j in range(len(w_overlaps)):
            bbox_list = []
            for cls in range(1, num_classes + 1):
                for idx, bbox in enumerate(results[key_name[i * w_len + j]][0][cls]):  # 0 represent 1 image
                    if bbox[8] > vis_thresh:
                        bbox[np.arange(0, 8, 2)] = bbox[np.arange(0, 8, 2)] + w_overlaps[j] * 512
                        bbox[np.arange(1, 8, 2)] = bbox[np.arange(1, 8, 2)] + h_overlaps[i] * 512
                        bbox_list = bbox_list + [np.append(np.append(bbox[0:8], bbox[8]), int(cls))]

            bbox_total[key_name[i * w_len + j]] = np.array(bbox_list)
            bbox_four_patch = copy.deepcopy(bbox_list)
            if i != 0:
                bbox_four_patch.extend(bbox_bottom[(i - 1) * w_len + j])  # add element and not add dim
            if j != 0:
                bbox_four_patch.extend(bbox_right[i * w_len + j - 1])
            # origin
            bbox_four_patch = np.array(bbox_four_patch)
            # area nms
            bbox_four_patch = area_and_point_nms(bbox_four_patch) if bbox_four_patch.shape[0] != 0 else np.array([])

            h_max = h_overlaps[i + 1] * 512 if i < h_len - 1 else h_overlaps[-1] * 512 + 512
            w_max = w_overlaps[j + 1] * 512 if j < w_len - 1 else w_overlaps[-1] * 512 + 512
            valid_tmp = []
            right_tmp = []
            bottom_tmp = []
            for _ in bbox_four_patch:
                x_max = _[0:8:2].max()
                y_max = _[1:8:2].max()
                if x_max < w_max and y_max < h_max:
                    valid_tmp.append(_)
                if x_max >= w_max:
                    right_tmp.append(_)
                if y_max >= h_max:
                    bottom_tmp.append(_)
            bbox_four.extend(valid_tmp)
            bbox_right[i * w_len + j] = right_tmp
            bbox_bottom[i * w_len + j] = bottom_tmp
            bbox_right_list.extend(right_tmp)
    return bbox_four


def ArraySwapLocation(bboxes):
    bboxes_modify = copy.deepcopy(bboxes[0])
    for i in range(1, 1 + num_classes):
        box = bboxes[0][i]
        flag = []
        if box.shape[0] == 0:
            continue
        for idx, box_s in enumerate(box):
            A = Point(box_s[2], box_s[3])
            B = Point(box_s[8], box_s[9])
            C = Point(box_s[4], box_s[5])
            D = Point(box_s[6], box_s[7])
            if is_intersected(A, B, C, D):
                flag.append(idx)
        if len(flag) == 0:
            continue
        box_tmp = box[np.array(flag)]
        # print('modify {}'.format(box_tmp[:,8]))
        for j, idx in enumerate(flag):
            bboxes_modify[i][idx][6] = box_tmp[j][8]
            bboxes_modify[i][idx][8] = box_tmp[j][6]
            bboxes_modify[i][idx][7] = box_tmp[j][9]
            bboxes_modify[i][idx][9] = box_tmp[j][7]
    return bboxes_modify


class Point(object):
    def __init__(self, x, y):
        self.x, self.y = x, y


class Vector(object):
    def __init__(self, start_point, end_point):
        self.start, self.end = start_point, end_point
        self.x = end_point.x - start_point.x
        self.y = end_point.y - start_point.y

def negative(vector):
    """negative"""
    return Vector(vector.end, vector.start)

def vector_product(vectorA, vectorB):
    """calculate x_1 * y_2 - x_2 * y_1"""
    return vectorA.x * vectorB.y - vectorB.x * vectorA.y

def is_intersected(A, B, C, D):
    """A, B, C, D is point class type"""
    AC = Vector(A, C)
    AD = Vector(A, D)
    BC = Vector(B, C)
    BD = Vector(B, D)
    CA = negative(AC)
    CB = negative(BC)
    DA = negative(AD)
    DB = negative(BD)

    return (vector_product(AC, AD) * vector_product(BC, BD) <= ZERO) \
        and (vector_product(CA, CB) * vector_product(DA, DB) <= ZERO)


def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    if len(opt.gpus) == 1:
        torch.cuda.set_device(opt.gpus[0])
    # initialization detector
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    if os.path.isdir(opt.demo):
        image_names = []
        ls = os.listdir(opt.demo)
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                image_names.append(os.path.join(opt.demo, file_name))
    else:
        image_names = [opt.demo]

    count = 0
    for (image_name) in image_names:
        count += 1
        results = {}
        key_name = []
        image_np = cv2.imread(image_name)
        image_show = copy.deepcopy(image_np)
        imgs, h_overlaps, w_overlaps, img_resolution = patchImg(image_np, resolution=1, scale=512, overlap=0.25)
        name = image_name.split('/')[-1].split('.')[0]
        # show rectangle
        # image_np = show_line(image_np, h_overlaps, w_overlaps, 512)
        # patch_num = len(h_overlaps)*len(w_overlaps)
        print('\r{}:{}'.format(count, name), end='')
        for i, img in enumerate(imgs):
            # cv2.imwrite('/home/lyc/data/dota_results/no_nms_show/{}_origin.jpg'.format(name + '_' + str(i)), img)
            img_ori = copy.deepcopy(img)
            start = time.time()
            scale_det, det = detector.run_dota(img_ori)
            print(time.time()-start)
            # det_swap = ArraySwapLocation(det)
            results[name + '_' + str(i)] = det
            key_name.append(name + '_' + str(i))
            # show_bbox(det_swap, num_classes, img_ori, '/home/lyc/data/dota_results/test/show/'+name + '_' + str(i) + '_cs.jpg')

        bbox_image = uniform_bbox_nms(h_overlaps, w_overlaps, key_name, results, 0.3)
        bbox_image_total = bbox_image
        # bbox_image_total = bbox_image_nms.astype('float32')
        # return nms results(list type)

        # bbox_image_total_idx = poly_nms_gpu(bbox_image_total, 0.1)
        # bbox_image_total = bbox_image_total[bbox_image_total_idx]

        # bbox_image_total = nms_overlap_all(bbox_image_total, 0.1, 0)
        # bbox_dict = bbox_txt_ready_nms(bbox_image_total, bbox_dict, name)
        name_save = '/Dataset/DOTA/show/show_jpg/{}.jpg'.format(name)
        show_bbox_angle(bbox_image_total, num_classes, image_show, name_save)
        # line_img = show_line(image_show, h_overlaps, w_overlaps)
        # cv2.imwrite('/home/lyc/data/dota_results/no_nms_show/line_{}.jpg'.format(name), line_img)
    # create_txt('/home/lyc/data/dota_results/txt/', bbox_dict)


if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
    # A = Point(0,0)
    # B = Point(1,0)
    # C = Point(1,1)
    # D = Point(0,1)
    # start = time.time()
    # print(is_intersected(A, B, C, D))