from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset
from .sample.ctdet_angle import CTDetDotaAngleDataset
from .sample.multi_dota_four import MultiPoseDotaFourDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP
from .dataset.coco_dota_angle import COCO_DOTA_ANGLE
from .dataset.coco_dota_four import COCO_DOTA_FOUR

dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP,
  'dota_angle': COCO_DOTA_ANGLE,
  'dota_four': COCO_DOTA_FOUR
}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset,
  'dota_angle': CTDetDotaAngleDataset,
  'dota_four': MultiPoseDotaFourDataset
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
