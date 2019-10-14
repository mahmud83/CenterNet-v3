from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math
import copy
import cv2

class MultiPoseDotaFourDataset(data.Dataset):
    def _calculate_wh(self, box):
        h = math.sqrt((box[0] - box[2]) ** 2 + (box[1] - box[3]) ** 2)
        w = math.sqrt((box[2] - box[4]) ** 2 + (box[3] - box[5]) ** 2)
        return h, w

    def _general_equation(self, point1, point2):
        x1 = point1[0]
        y1 = point1[1]
        x2 = point2[0]
        y2 = point2[1]
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        return A, B, C

    def _calculate_intersection_point(self, box):
        # intersect two lines, calculate center point
        box_four = box.reshape(-1, 2)
        A1, B1, C1 = self._general_equation(box_four[0], box_four[2])
        A2, B2, C2 = self._general_equation(box_four[1], box_four[3])
        m = A1 * B2 - A2 * B1
        if m == 0:
            return np.array([sum([box_four[i][0] for i in range(4)]) / 4, sum([box_four[i][1] for i in range(4)]) / 4],
                            dtype=np.float32)
        else:
            x = (C2 * B1 - C1 * B2) / m
            y = (C1 * A2 - C2 * A1) / m
        return np.array([x, y], dtype=np.float32)

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)
        img_show = copy.deepcopy(img)

        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        rot = 0

        # flipped = False
        # if self.split == 'train':
        #     if not self.opt.not_rand_crop:
        #         s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        #         w_border = self._get_border(128, img.shape[1])
        #         h_border = self._get_border(128, img.shape[0])
        #         c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        #         c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
        #     else:
        #         sf = self.opt.scale
        #         cf = self.opt.shift
        #         c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
        #         c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
        #         s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        #     if np.random.random() < self.opt.aug_rot:
        #         rf = self.opt.rotate
        #         rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)

        trans_input = get_affine_transform(
            c, s, rot, [self.opt.input_res, self.opt.input_res])
        inp = cv2.warpAffine(img, trans_input,
                             (self.opt.input_res, self.opt.input_res),
                             flags=cv2.INTER_LINEAR)
        ################## plot
        # cv2.imwrite('/Workspace/CenterNet/in_{}'.format(file_name), inp)

        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        # ----------------------------------------- inp finished
        output_res = self.opt.output_res
        self.num_joints = self.opt.num_joints
        num_joints = self.num_joints
        trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])
        trans_output = get_affine_transform(c, s, 0, [output_res, output_res])
        ################# plot
        # inp_out = cv2.warpAffine(img_show, trans_output,
        #                          (output_res, output_res),
        #                          flags=cv2.INTER_LINEAR)
        # for k in range(num_objs):
        #     ann = anns[k]
        #     bbox = copy.deepcopy(ann['bbox'])
        #     bbox[:2] = affine_transform(bbox[:2], trans_output)
        #     bbox[2:4] = affine_transform(bbox[2:4], trans_output)
        #     bbox[4:6] = affine_transform(bbox[4:6], trans_output)
        #     bbox[6:8] = affine_transform(bbox[6:8], trans_output)
        #
        #     bbox = np.clip(bbox, 0, output_res - 1)
        #     ct = self._calculate_intersection_point(bbox)
        #     ct_int = ct.astype(np.int32)
        #     # countour = cv2.boxPoints(((bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox[4] / math.pi * 180))
        #     # cv2.drawContours(inp_out, [np.array(bbox).reshape(4,2).astype(int)], 0, (0, 0, 255), 2)
        #     cv2.circle(inp_out, tuple(ct_int), 2, (0, 0, 255), -1)
        # print('file {} num  {}'.format(file_name, num_objs))
        # cv2.imwrite('/Workspace/CenterNet/out_{}'.format(file_name), inp_out)

        hm = np.zeros((self.num_classes, output_res, output_res), dtype=np.float32)
        hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
        dense_kps = np.zeros((num_joints, 2, output_res, output_res),
                             dtype=np.float32)
        dense_kps_mask = np.zeros((num_joints, output_res, output_res),
                                  dtype=np.float32)
        kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)
        hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)
        hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)
        hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = np.array(ann['bbox'])
            cls_id = int(ann['category_id']) - 1
            pts = np.array(ann['segmentation'], np.float32).reshape(num_joints, 2)

            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:4] = affine_transform(bbox[2:4], trans_output)
            bbox[4:6] = affine_transform(bbox[4:6], trans_output)
            bbox[6:8] = affine_transform(bbox[6:8], trans_output)

            bbox = np.clip(bbox, 0, output_res - 1)

            h, w = self._calculate_wh(bbox)
            if (h > 0 and w > 0) or (rot != 0):
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius))
                ct = self._calculate_intersection_point(bbox)
                ct_int = ct.astype(np.int32)

                ind[k] = ct_int[1] * output_res + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1

                hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                hp_radius = self.opt.hm_gauss \
                    if self.opt.mse_loss else max(0, int(hp_radius))
                for j in range(num_joints):
                    # if pts[j, 2] > 0:
                    if True:
                        pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
                        if pts[j, 0] >= 0 and pts[j, 0] < output_res and \
                                        pts[j, 1] >= 0 and pts[j, 1] < output_res:
                            kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                            kps_mask[k, j * 2: j * 2 + 2] = 1
                            pt_int = pts[j, :2].astype(np.int32)
                            hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
                            hp_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]
                            hp_mask[k * num_joints + j] = 1
                            if self.opt.dense_hp:
                                # must be before draw center hm gaussian
                                draw_dense_reg(dense_kps[j], hm[cls_id], ct_int,
                                               pts[j, :2] - ct_int, radius, is_offset=True)
                                draw_gaussian(dense_kps_mask[j], ct_int, radius)
                            draw_gaussian(hm_hp[j], pt_int, hp_radius)
                draw_gaussian(hm[cls_id], ct_int, radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1] +
                              pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])
        if rot != 0:
            hm = hm * 0 + 0.9999
            reg_mask *= 0
            kps_mask *= 0
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind,
               'hps': kps, 'hps_mask': kps_mask}
        if self.opt.dense_hp:
            dense_kps = dense_kps.reshape(num_joints * 2, output_res, output_res)
            dense_kps_mask = dense_kps_mask.reshape(
                num_joints, 1, output_res, output_res)
            dense_kps_mask = np.concatenate([dense_kps_mask, dense_kps_mask], axis=1)
            dense_kps_mask = dense_kps_mask.reshape(
                num_joints * 2, output_res, output_res)
            ret.update({'dense_hps': dense_kps, 'dense_hps_mask': dense_kps_mask})
            del ret['hps'], ret['hps_mask']
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.hm_hp:
            ret.update({'hm_hp': hm_hp})
        if self.opt.reg_hp_offset:
            ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 40), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta
        return ret
