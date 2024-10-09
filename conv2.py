
import numpy as np

from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel

from reproject import reproject_exact,reproject_adaptive
from astropy.wcs import WCS

from matplotlib import pyplot as plt
from astropy import units as u
from astropy.visualization import quantity_support
quantity_support()  
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus


lowcolor = 'steelblue'
highcolor='crimson'

def convolve_resamp(data_lr,data_hr,info_lr,info_hr):
    '''
    convolves and resamples
    '''

    bmaj_pix = info_lr['bmaj']/(info_hr['pix_size'][0].value)
    bmin_pix = info_lr['bmin']/(info_hr['pix_size'][0].value)

    print("bmajpix, bminpix: ",bmaj_pix,bmin_pix)

    bmaj_sigma = bmaj_pix/(2*np.sqrt(2*np.log(2)))
    bmin_sigma = bmin_pix/(2*np.sqrt(2*np.log(2)))

    print("bmajsigma, bminsigma: ",bmaj_sigma,bmin_sigma)

    gaussian_2D_kernel = Gaussian2DKernel(x_stddev=bmaj_sigma, y_stddev=bmin_sigma, theta = info_lr['theta'], mode = 'oversample')

    convolved_hr_pix = convolve(data_hr['jy/pixel'][0,0], gaussian_2D_kernel) 
    print("convolution complete.")

    beam_solid_angle_hr = np.pi * info_hr['beam'][0] * info_hr['beam'][1] / (4*np.log(2))
    beam_solid_angle_lr = np.pi * info_lr['beam'][0] * info_lr['beam'][1] / (4*np.log(2))

    #convolved_hr_jy_arc2 = convolved_hr_pix / beam_solid_angle_hr.to(u.arcsec**2).value

    wcs_hr = WCS(info_hr['header'])
    wcs_lr = WCS(info_lr['header'])

    print("starting reprojection...")

    #reproj_hr_pix, footprint = reproject_exact((convolved_hr_pix[np.newaxis,np.newaxis], wcs_hr.celestial), wcs_lr.celestial)
    reproj_hr_pix, footprint = reproject_adaptive((convolved_hr_pix[np.newaxis,np.newaxis], wcs_hr.celestial), wcs_lr.celestial,
                                conserve_flux=True)
    reproj_hr_jy_arc2 =(reproj_hr_pix /beam_solid_angle_lr.to(u.arcsec**2).value)

    reproj_hr_pixel_norm = reproj_hr_pix*np.nanmax(data_lr['jy/pixel'])/np.nanmax(reproj_hr_pix) #normalizing
    reproj_hr_jy_arc2_norm =(reproj_hr_pixel_norm /beam_solid_angle_lr.to(u.arcsec**2).value)

    print("data_hr: ",np.nansum(data_hr['jy/pixel']))
    print("convovled_hr_pix: ",np.nansum(convolved_hr_pix))

    print("reproj_hr_pix: ",np.nansum(reproj_hr_pix))
    print("reproj_hr_pix_norm: ",np.nansum(reproj_hr_pixel_norm))
    print()
    print("reproj_hr_arc: ",np.nansum(reproj_hr_jy_arc2))
    print("reproj_hr_arc_norm: ",np.nansum(reproj_hr_jy_arc2_norm))


    max_pos = np.unravel_index(np.argmax(np.ma.masked_invalid(reproj_hr_pixel_norm[0,0,...])), reproj_hr_pixel_norm.shape)
    y = max_pos[-2]
    x = max_pos[-1]

    position_conv = (x,y)

    ### GATHERING MEDIAN BACKGROUND DATA
    mean_cv, median_cv, std_cv = sigma_clipped_stats(reproj_hr_pixel_norm[0,0,...], sigma = 3.0)
    # print("Mean, meadian, std: ", mean, median, std)

    daofind = DAOStarFinder(fwhm = 3.0, threshold = 5*std_cv.value)
    print(type(median_cv),type(reproj_hr_pixel_norm))
    sources = daofind(reproj_hr_pixel_norm[0,0,...] - median_cv)
    for col in sources.colnames:
        if col not in ('id', 'npix'):
            sources[col].info.format = '%.2f' # for table formatting
    sources.pprint(max_width=-1)

    y=int(info_lr['position'][1])
    x=int(info_lr['position'][0])
    w=200
    data_lr_centered = data_lr['jy/pixel'][0,0,y-w:y+w,x-w:x+w]
    reproj_hr_pixel_centered = reproj_hr_pixel_norm[0,0,position_conv[1]-w:position_conv[1]+w,position_conv[0]-w:position_conv[0]+w]


    csm = data_lr_centered - (reproj_hr_pixel_centered)

    
    csm_jy_arc = csm/(beam_solid_angle_lr.to(u.arcsec**2).value)

    # plt.plot(data_lr_centered, label = 'lr', c = lowcolor)
    # plt.plot(reproj_hr_pixel_centered, label = 'hrconv', c='maroon')
    # plt.plot(csm, 'o-', label='CSM', zorder=9,c='purple')
    # plt.legend()

    # plt.xlim(0,1.5)
    # plt.semilogy()
    # plt.show()

    print("stop 2")

    plt.imshow(csm.value, origin = 'lower')
    plt.title("CSM")
    plt.plot(200, 200,'rx')
    plt.plot(info_hr['position'][0],info_hr['position'][1],'o')
    plt.colorbar()
    plt.xlim(180,220)
    plt.ylim(180,220)
    plt.show()

    print("stop 3")
    
    data_conv_norm = {'jy/arc2': reproj_hr_jy_arc2_norm, 'jy/pixel': reproj_hr_pixel_norm}
    data_conv = {'jy/arc2': reproj_hr_jy_arc2,'jy/pixel': reproj_hr_pix}
    data_centered = {'lr': data_lr_centered,'convhr':reproj_hr_pixel_centered}
    data_csm = {'jy/arc2':csm_jy_arc, 'jy/pix': csm}

    info_conv = {'position': position_conv, 'median': median_cv}


    return data_conv_norm, data_conv, data_centered, data_csm, info_conv
    
    