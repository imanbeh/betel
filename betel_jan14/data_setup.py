#importing all of the data we already calculated in setup
# + importing reconv functions

from conv_reproj import match
from header_setup import data_hr, data_lr, info_hr, info_lr, datas
from align_and_diff import a_d

from matplotlib import pyplot as plt
import numpy as np


## Reprojection
data_reproj, info_reproj = match(data_hr['jy_pix'],data_lr['jy_pix'], info_hr, info_lr)

data_csm = a_d(data_reproj['jy_pix_norm'], data_lr['jy_pix'], info_reproj,info_lr)