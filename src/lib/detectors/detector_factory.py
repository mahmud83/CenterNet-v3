from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .exdet import ExdetDetector
from .ddd import DddDetector
from .ctdet import CtdetDetector
from .multi_pose import MultiPoseDetector
from .ctdet_angle import CtdetAngleDetector
from .multi_dota_four import CtdetFourDetector

detector_factory = {
  'exdet': ExdetDetector, 
  'ddd': DddDetector,
  'ctdet': CtdetDetector,
  'multi_pose': MultiPoseDetector,
  'dota_angle': CtdetAngleDetector,
  'dota_four': CtdetFourDetector
}
