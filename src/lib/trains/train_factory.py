from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetTrainer
from .ddd import DddTrainer
from .exdet import ExdetTrainer
from .multi_pose import MultiPoseTrainer
from .ctdet_angle import CtdetAngleTrainer
from .multi_dota_four import MultiPoseDotaFourTrainer

train_factory = {
  'exdet': ExdetTrainer, 
  'ddd': DddTrainer,
  'ctdet': CtdetTrainer,
  'multi_pose': MultiPoseTrainer,
  'dota_angle': CtdetAngleTrainer,
  'dota_four': MultiPoseDotaFourTrainer
}
