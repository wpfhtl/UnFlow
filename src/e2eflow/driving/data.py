import os
import sys

import numpy as np
import matplotlib.image as mpimg

from ..core.data import Data
from ..util import tryremove


class DrivingData(Data):
    # DRIVING_URL = 'https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Driving/raw_data/driving__frames_finalpass.tar' # download dataset from Internet

    # dirs = ['driving_frames_finalpass']

    def __init__(self, data_dir, stat_log_dir=None,
                 development=True, fast_dir=None):
        super().__init__(data_dir, stat_log_dir,
                         development=development,
                         fast_dir=fast_dir)

    # def _fetch_if_missing(self):
    #     local_path = os.path.join(self.data_dir, 'driving_frames_finalpass')
    #     if not os.path.isdir(local_path):
    #         self._download_and_extract(self.DRIVING_URL, local_path)

    def get_raw_dirs(self):
        dirs = []
        for folder in ['fine_tune_test', 'fine_tune_training', 'test', 'training']:
        # for folder in ['15mm_focallength/scene_backwards', '15mm_focallength/scene_forwards','35mm_focallength/scene_backwards', '35mm_focallength/scene_forwards']:
            top_dir = os.path.join(self.current_dir, 'data_sf/driving_frames_finalpass/' + folder)
            for sub_dir in os.listdir(top_dir):
              dirs.append(os.path.join(top_dir, sub_dir))
        return dirs
