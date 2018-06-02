# This script produces the final submission as shown on
# the kaggle leaderboard.
#
# It runs correctly if placed next to the folders /src
# and /data. The folder /src contains whatever other
# scripts you need (provided by you). The folder /data
# can be assumed to contain two folders /set_train and
# /set_test which again contain the training and test
# samples respectively (provided by user, i.e. us).
#
# Its output is "final_sub.csv"

import inspect
import os
import sys
from os.path import join

curr_folder = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 
parent_data_folder = join(curr_folder, 'data/')
src_folder = join(curr_folder, 'src/')
sys.path.insert(0, src_folder)

from classify3 import run_classify
from segment import run_segment

run_segment('train', parent_data_folder)
run_segment('test', parent_data_folder)
run_classify(curr_folder)