'''
aligns and differences after reprojection and convolution
adds rows instead of subtracting them
'''

import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u


def a_d(reproj_hr_pixel_norm, data_jy_per_pixel_lr,info_conv,info_lr):
    '''
    reproj_hr_pixel_norm, data_jy_per_pixel_lr,info_conv,info_lr
    '''


    # x_lr=int(info_lr['position'][0])
    # y_lr=int(info_lr['position'][1])

    # x_hr = info_conv['position'][0]
    # y_hr = info_conv['position'][1]
    #data_jy_per_pixel_lr[0,0,x_lr,y_lr]
    #add extra rows and columns to center

    data_lr_centered_add = np.insert(data_jy_per_pixel_lr[0,0,...], (0,0,0,0), np.nan, axis=0)
    data_lr_centered_add = np.insert(data_lr_centered_add, (0), np.nan, axis=1)
    data_lr_centered_add = np.insert(data_lr_centered_add, (data_lr_centered_add.shape[1],data_lr_centered_add.shape[1],data_lr_centered_add.shape[1]), np.nan, axis=1)
    
    reproj_hr_pixel_centered_add = np.insert(reproj_hr_pixel_norm[0,0,...], (0,0,0,0), np.nan, axis=1)
    reproj_hr_pixel_centered_add = np.insert(reproj_hr_pixel_centered_add, (0,0,0), np.nan, axis=0)
    reproj_hr_pixel_centered_add = np.insert(reproj_hr_pixel_centered_add, (reproj_hr_pixel_centered_add.shape[0]), np.nan, axis=0)
    
    csm_jy_pixel = data_lr_centered_add-reproj_hr_pixel_centered_add
    csm_jy_arc = csm_jy_pixel/info_lr['beam_solid_angle'].to(u.arcsec**2).value


    data_csm = {'jy_pix': csm_jy_pixel, 'jy_arc2': csm_jy_arc}

    return data_csm

