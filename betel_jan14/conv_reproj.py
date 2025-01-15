import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u


def match(data_jy_per_pixel_hr, data_jy_per_pixel_lr, info_hr, info_lr):
    '''
        data 1, data 2 in jy/pix, the info dictionaries
    '''

    # Convolution
    from astropy.convolution import convolve
    from astropy.convolution import Gaussian2DKernel

    from reproject import reproject_adaptive
    bmaj_pix = info_lr['bmaj']/(info_hr['pix_size'][0].value)
    bmin_pix = info_lr['bmin']/(info_hr['pix_size'][0].value)

    bmaj_sigma = bmaj_pix/(2*np.sqrt(2*np.log(2)))
    bmin_sigma = bmin_pix/(2*np.sqrt(2*np.log(2)))

    gaussian_2D_kernel = Gaussian2DKernel(x_stddev=bmaj_sigma, y_stddev=bmin_sigma, theta = info_lr['theta'], mode = 'oversample')

    convolved_hr_pix = convolve(data_jy_per_pixel_hr[0,0], gaussian_2D_kernel) 
    print("convolution complete.")

    beam_solid_angle_hr = np.pi * info_hr['beam'][0] * info_hr['beam'][1] / (4*np.log(2))
    beam_solid_angle_lr = np.pi * info_lr['beam'][0] * info_lr['beam'][1] / (4*np.log(2))

    from astropy.wcs import WCS
    wcs_hr = WCS(info_hr['header'])
    wcs_lr = WCS(info_lr['header'])
    print("starting reprojection...")

    #reproj_hr_pix, footprint = reproject_exact((convolved_hr_pix[np.newaxis,np.newaxis], wcs_hr.celestial), wcs_lr.celestial)
    reproj_hr_pix, footprint = reproject_adaptive((convolved_hr_pix[np.newaxis,np.newaxis], wcs_hr.celestial), wcs_lr.celestial,
                                conserve_flux=True)

    print("reprojection complete.")

    reproj_hr_jy_arc2 =(reproj_hr_pix /beam_solid_angle_lr.to(u.arcsec**2).value)

    reproj_hr_pixel_norm = reproj_hr_pix*np.nanmax(data_jy_per_pixel_lr[0,0])/np.nanmax(reproj_hr_pix) #normalizing
    reproj_hr_jy_arc2_norm =(reproj_hr_pixel_norm /beam_solid_angle_lr.to(u.arcsec**2).value)

    max_pos = np.unravel_index(np.argmax(np.ma.masked_invalid(reproj_hr_pixel_norm[0,0,...])), reproj_hr_pixel_norm.shape)
    y = max_pos[-1]
    x = max_pos[-2]

    position_conv = (x,y)

    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder
    ### GATHERING MEDIAN BACKGROUND DATA
    mean_cv, median_cv, std_cv = sigma_clipped_stats(reproj_hr_pixel_norm[0,0,...], sigma = 3.0)
    # print("Mean, meadian, std: ", mean, median, std)

    daofind = DAOStarFinder(fwhm = 3.0, threshold = 5*std_cv.value)

    data_reproj = {"jy_arc2": reproj_hr_jy_arc2, "jy_pix": reproj_hr_pix, "jy_pix_norm": reproj_hr_pixel_norm,
              "jy_arc2_norm": reproj_hr_jy_arc2_norm}
    info_reproj = {"position": position_conv, "median": median_cv}

    return data_reproj, info_reproj

