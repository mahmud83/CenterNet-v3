import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPoint
import shapely
import copy
from nms_iou import polyiou
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
# dota_class_name = ['__background__', 'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
#                 'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter'
#             ]
dota_class_name = ['__background__', 'small-vehicle', 'large-vehicle']
nms = None  # not needed in pascal evaluation

class GaussDistribution():
    def __init__(self, mu, Sigma):
        self.mu = mu
        self.sigma = Sigma

    def tow_d_gaussian(self, x):
        mu = self.mu
        Sigma = self.sigma
        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2 * np.pi) ** n * Sigma_det)
        fac = np.einsum('...k,kl,...l->...', x - mu, Sigma_inv, x - mu)
        return np.exp(-fac / 2) / N

    def one_d_gaussian(self, x):
        mu = self.mu
        sigma = self.sigma
        N = np.sqrt(2 * np.pi * np.power(sigma, 2))
        fac = np.power(x - mu, 2) / np.power(sigma, 2)
        return np.exp(-fac / 2) / N

    def one_gaussian_new(self, x):
        mu = self.mu
        sigma = self.sigma
        N = np.sqrt(2 * np.pi * np.power(sigma, 2))
        fac = np.power(x - mu, 2) / np.power(sigma, 2)
        return np.exp(-fac / 2)


def add_bbox(img, bbox, cat, conf=1, show_txt=True):
    # center = bbox[0:2].astype('int')
    bbox = bbox[0:8]
    bbox = np.array(bbox, dtype=np.int32)
    bbox = np.reshape(bbox, newshape=(4, 2))
    # cv2.drawContours(inp_out, [bbox.astype(int)], 0, (0, 0, 255), 2)
    # cat = (int(cat) + 1) % 80
    cat = int(cat)
    # print('cat', cat, self.names[cat])
    colors = [(color_list[_]).astype(np.uint8) \
            for _ in range(len(color_list))]
    colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
    c = colors[cat][0][0].tolist()
    txt = '{}{:.1f}'.format(dota_class_name[cat], conf)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    # cv2.circle(img, tuple(center), 5, (0, 0, 255), -1)
    # draw circle to distinguish whether clock wise
    # for i, bbox_single in enumerate(bbox):
    # for bbox_s in bbox:
    #     cv2.circle(img, tuple(bbox_s), 3, (0, 0, 255), -1)
    #     cv2.putText(img, str(i), tuple(bbox_single), font, 0.5, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    cv2.drawContours(img, [bbox.astype(int)], 0, c, 2)
    if show_txt:
        cv2.rectangle(img,
                      (bbox[0][0], bbox[0][1] - cat_size[1] - 2),
                      (bbox[0][0] + cat_size[0], bbox[0][1] - 2), c, -1)
        cv2.putText(img, txt, (bbox[0][0], bbox[0][1] - 2),
                    font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    return img

def polygon_iou(np1, np2):
    """
    Intersection over union between two shapely polygons.
    """
    polygon_points1 = np1.reshape(4, 2)
    poly1 = Polygon(polygon_points1).convex_hull
    polygon_points2 = np2.reshape(4, 2)
    poly2 = Polygon(polygon_points2).convex_hull
    union_poly = np.concatenate((polygon_points1, polygon_points2))
    if not poly1.intersects(poly2):  # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                return 0
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

def nms_overlap(bboxes, nms_thresh):
    bbox_new = copy.deepcopy(bboxes)
    num_classes = len(bboxes)
    for cls_ind in range(num_classes):
        if bboxes[cls_ind].shape[0] == 0:
            continue
        # coordinate = bboxes[cls_ind][:, 0:8]
        score = bboxes[cls_ind][:, -1]
        order = score.argsort()[::-1]  # reverse vector
        boxes = bboxes[cls_ind]
        out_boxes = []
        for i in range(boxes.shape[0]):
            box_i = boxes[order[i]]
            if box_i[8] > 0:
                out_boxes.append(box_i)
                for j in range(i + 1, len(boxes)):
                    box_j = boxes[order[j]]
                    if polygon_iou(box_i[0:8], box_j[0:8]) > nms_thresh:
                        # print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                        box_j[8] = 0
        bbox_new[cls_ind] = np.array(out_boxes)
    return bbox_new

def nms_overlap_all(bboxes, nms_thresh, area_threshold):
    def area_filter(bbox):
        area = polyiou.area_value(polyiou.VectorDouble(bbox[2:10].astype(float)))
        if area > area_threshold:
            return True
        return False
    # if bboxes
    # print(bboxes.shape)
    score = bboxes[:, -2]
    # print(bboxes.shape)
    order = score.argsort()[::-1]  # reverse vector
    out_boxes = []
    count = 0
    for i in range(bboxes.shape[0]):
        box_i = bboxes[order[i]]
        if box_i[10] > 0:
            # if not area_filter(box_i):  # filter area
            #     box_i[10] = 0
            #     continue
            out_boxes.append(box_i)
            for j in range(i + 1, len(bboxes)):
                count += 1
                box_j = bboxes[order[j]]
                # if not area_filter(box_j):  # filter area
                #     box_j[10] = 0
                #     continue
                overlap = polyiou.iou_poly(polyiou.VectorDouble(box_i[2:10].astype(float)),
                                           polyiou.VectorDouble(box_j[2:10].astype(float)))
                if overlap > nms_thresh:
                    # print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[10] = 0

    bbox_new = np.array(out_boxes)
    return bbox_new, count

def soft_nms_overlap_all(bboxes, nms_thresh):
    score = bboxes[:, -2]
    order = score.argsort()[::-1]  # reverse vector
    out_boxes = []
    for i in range(bboxes.shape[0]):
        box_i = bboxes[order[i]]
        if box_i[8] > 0:
            out_boxes.append(box_i)
            for j in range(i + 1, len(bboxes)):
                box_j = bboxes[order[j]]
                iou_v = polygon_iou(box_i[0:8], box_j[0:8])
                # if iou_v > nms_thresh:
                #     box_j[8] = 0
                if iou_v > nms_thresh:
                    box_j[8] = box_j[8] * np.exp(-(iou_v * iou_v) / 0.5)
                    if box_j[8] < 0.3:
                        box_j[8] = 0

    bbox_new = np.array(out_boxes)
    return bbox_new

def PointInBbox(box, point):
    corner = np.append(box[0:8], box[0:2]).reshape(-1, 2)
    area_all = abs(polyiou.area_value(polyiou.VectorDouble(box[0:8])))
    # area_all = Polygon(corner).convex_hull.area
    corner_center = []
    for i in range(4):
        corner_center.append(list(corner[i]) + list(corner[i + 1]) + point + list(corner[i]))
    # area = [Polygon(np.array(center_s).reshape(-1, 2)).convex_hull.area for center_s in corner_center]
    area = [abs(polyiou.area_value(polyiou.VectorDouble(np.array(center_s)))) for center_s in corner_center]
    return sum(area) == area_all

def calarea(bbox, img):
    # img_show = copy.deepcopy(img)
    flag = []
    for i in range(1, 1 + num_classes):
        bboxes = bbox[i]
        if len(bboxes) == 0:
            continue
        # for j in range(bboxes.shape[0]):
        area1 = [abs(polyiou.area_value(polyiou.VectorDouble(bboxes[j][2:10].astype(float)))) for j in range(bboxes.shape[0])]
        area = [abs(Polygon(bboxes[j][2:10].reshape(-1, 2)).convex_hull.area) for j in range(bboxes.shape[0])]
        flag.append(area == area1)
        # for idx, box in enumerate(bboxes):
        #     if idx in flag:
        #         cv2.putText(img_show, str(idx), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        #     cv2.drawContours(img_show, [box[2:10].astype(int).reshape(-1,2)], 0, (0, 0, 255), 2)
        # cv2.imwrite('/home/lyc/data/dota_results/no_nms_show/{}_area.jpg'.format(i), img_show)
    if False in flag:
        return False
    else:
        return True

def area_and_point_nms(bbox):
    # nms each class
    bbox_dict = {}
    for idx, box in enumerate(bbox):
        cls = int(box[-1])
        bbox_dict[cls] = bbox_dict.get(cls, []) + [box]
    bbox_return = []
    for key, value in bbox_dict.items():
        bboxes = np.array(value)
        area = [abs(polyiou.area_value(polyiou.VectorDouble(bboxes[i][0:8]))) for i in range(bboxes.shape[0])]
        # area = [Polygon(bboxes[i][2:10].reshape(-1, 2)).convex_hull.area for i in range(bboxes.shape[0])]

        center_point = list(map(list, bboxes[:, 0:2]))
        area = np.array(area)
        order = area.argsort()[::-1]  # reverse vector
        out_boxes = []
        for i in range(bboxes.shape[0]):
            box_i = bboxes[order[i]]
            if area[order[i]] < 100:
                continue
            if box_i[8] > 0:
                out_boxes.append(box_i)
                # SL_i = CalSideLen(box_i)
                for j in range(i + 1, len(bboxes)):
                    box_j = bboxes[order[j]]
                    if area[order[j]] < 100:
                        continue
                    center_j = center_point[order[j]]
                    overlap = polyiou.iou_poly(polyiou.VectorDouble(box_i[0:8].astype(float)),
                                               polyiou.VectorDouble(box_j[0:8].astype(float)))
                    # overlap = polygon_iou(box_i[2:10], box_j[2:10])
                    if overlap > 0.1:
                        box_j[8] = 0
                        continue

                    if PointInBbox(box_i[0:8], center_j):
                        # in box directly
                        box_j[8] = 0
                        ###### in box and calculate point distance ######
                        # dist = []
                        # # calculate point distance
                        # for corner_point in box_j[0:8].reshape(-1, 2):
                        #     corner_point_peat = np.repeat(corner_point.reshape(1,2), 4, 0)
                        #     # calculate relative distance
                        #     for SL in SL_i:
                        #         dist_s = (((corner_point_peat - box_i[0:8].reshape(-1, 2)) ** 2).sum(1) ** 0.5) / SL < 0.2
                        #         dist.extend(list(dist_s))
                        # if True in dist:
                        #     box_j[10] = 0
        bbox_new = np.array(out_boxes)
        bbox_return.extend(bbox_new)

    return np.array(bbox_return)


def show_bbox(bbox_total, num_classes, image_np, name):
    # for bbox_total_s in bbox_total:
    image_show = copy.deepcopy(image_np)
    for j in range(1 , 1 + num_classes):
        for bbox in bbox_total[j]:
            if bbox[10] > 0.3:
                image_show = add_bbox(image_np, bbox[0:10], j + 1, bbox[10])
    cv2.imwrite(name, image_show)

def show_bbox_nms(bbox_total, num_classes, image_np, name):
    # for bbox_total_s in bbox_total:
    image = image_np

    for bbox in bbox_total:
        if bbox[8] > 0.3:
            image = add_bbox(image_np, bbox[0:8], int(bbox[9]), bbox[8])
            # image = add_bbox(image_np, bbox[:8], int(bbox[9]), bbox[8])
    cv2.imwrite(name, image)

if __name__ == '__main__':
    # bbox = np.array([0,0,0,1,1,1,1,0]).astype(float)
    bbox = np.array([0.5,0,1,1,0.5,0.5,0,1]).astype(float)
    center = [0.5, 0.25]
    print(PointInBbox(bbox, center))