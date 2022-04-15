# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 17:20:50 2022

@author: tripa
"""

import os
import numpy as np
from utils import mean_average_precision

res_dir = os.path.join(os.getcwd(), 'detection_results')
gt_bboxes = np.load('gt_results_1100.npy')
total_final_boxes = np.load('pred_results_1100.npy')

print('Ground-truth Boxes: {}, Predicted Boxes: {}'.format(len(gt_bboxes), len(total_final_boxes)))
print('Mean Average Precision: {:.4f}'.format(mean_average_precision(total_final_boxes, gt_bboxes, iou_threshold=0.1, num_classes=20, save=True, loc=res_dir)))

# %%
import math
window_size = 224
for aspect_ratio in [1, 2, 3, 0.5, 0.33]:
    window_height = int(window_size * math.sqrt(aspect_ratio))
    window_width = int(window_size / math.sqrt(aspect_ratio))
    print(window_height, window_width)