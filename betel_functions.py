#
# One nice big python file with all of our functions!
#

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
import cmasher as cmr
from astropy.modeling.models import BlackBody
from astropy.io import fits
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.visualization import quantity_support
quantity_support()  

from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus

from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel

from reproject import reproject_exact
from astropy.wcs import WCS

r_betelgeuse = 29.50*1e-3*u.arcsec
beta= 1.4
k160 = 8.8*u.cm**2 *u.g**-1
stellar_radius_hr_pc = (0.035*168/206265)*u.parsec
rhr = stellar_radius_hr_pc #pc hr radius
lmda = 887.0*u.um #microns
lmda_cm = 0.0887*u.cm
lmda_AA = 8.87e6*u.AA

lowcolor = 'steelblue'
highcolor='crimson'

#defining constants for density model

def sm(dist, ang_size):
    '''
    takes ang size arsec
    distance pc
    returns diam in pc
    '''
    diam = (ang_size*dist)/206265

    return diam

def rp_annulus(data, pix_size_arcsec,position):
    '''
    data, pix size in arcsec, position of center star
    '''
    aperture_size = r_betelgeuse.value / pix_size_arcsec.value #put in pixel units
    aperture = CircularAperture(position,aperture_size)
    photometry = aperture_photometry(data,aperture)
    # print()
    # print("Photometry: ", photometry)

    aperture_sizes = np.arange(1,200,1)
    annuli=[CircularAnnulus(position, r_in=aperture_sizes[i], r_out=aperture_sizes[i+1]) for i in range(len(aperture_sizes)-1)]
    areas = np.array([circle.area for circle in annuli])*pix_size_arcsec**2
    # print(apertures)
    
    
    photometry = aperture_photometry(data, annuli)
    phot_list = []
    for k,v in photometry.items():
        if 'aperture_sum' in k:
            phot_list.append(v)
    phot_array = np.array(phot_list)
    #print(phot_list)
    surf_brightness_jy_arc2 = phot_array.flatten()*u.Jy / (areas)

    centers_arc = ((aperture_sizes[1:]+aperture_sizes[:-1])/2)*pix_size_arcsec
    centers_pc = sm(168,centers_arc).value*u.pc

    centers = {'pc': centers_pc, 'arc': centers_arc}

    return centers,surf_brightness_jy_arc2


def radial_read(filename,w,x2=0,y2=0):
    '''
    reads in file and returns radial profile and returns important data
    '''

    hdu_hr = fits.open(filename, mode= 'update')
    data_hr = hdu_hr[0].data
    header_hr = hdu_hr[0].header

    theta = header_hr['BPA']
    bmaj = header_hr['BMAJ']
    bmin = header_hr['BMIN']

    ### CONVERTING DATA TO DIFFERENT UNITS
    beam = (header_hr['BMAJ'] * u.deg, header_hr['BMIN']*u.deg)
    pix_size = (np.abs(header_hr['CDELT1'])*u.deg,np.abs(header_hr['CDELT2'])*u.deg)
    pix_size_arcsec = pix_size[0].to(u.arcsec)
    beam_solid_angle = np.pi * beam[0] * beam[1] / (4*np.log(2))

    beam_solid_angle_arcsec2 = beam_solid_angle.to(u.arcsec**2).value # dividing by arcsec conversion to get from beam to arcsec
    pixels_per_beam = beam_solid_angle / (pix_size[0]*pix_size[1])

    data_jy_per_pixel = data_hr / pixels_per_beam

    ### GATHERING MEDIAN BACKGROUND DATA
    mean, median, std = sigma_clipped_stats(data_jy_per_pixel[0,0,...], sigma = 3.0)
    median_jy_per_arc2 = median /beam_solid_angle_arcsec2
    daofind = DAOStarFinder(fwhm = 3.0, threshold = 5*std.value)
    sources = daofind(data_jy_per_pixel[0,0,...] - median.value)
    for col in sources.colnames:
        if col not in ('id', 'npix'):
            sources[col].info.format = '%.2f' # for table formatting

    data_jy_per_pixel-=median.value
    data_jy_per_arcsec2 = data_jy_per_pixel / beam_solid_angle_arcsec2


    ### PHOTOMETRY CALCUATION
    max_pos = np.unravel_index(np.argmax(np.ma.masked_invalid(data_jy_per_pixel)), data_jy_per_pixel.shape)
    y = max_pos[-2]
    x = max_pos[-1]
    
    
    position = (x,y) #max_pos[-2], max[-1]
    if(x2!=0 and y2!=0):
        position = (x2,y2)

    centers, rp = rp_annulus(data_jy_per_pixel[0,0,...],pix_size[0].to(u.arcsec),position)

    ### RADIUS DATA IN 2D
    dimensions = data_jy_per_arcsec2[0,0,...].shape
    rows,columns = dimensions
    radius_2d = np.array([[0.0]*columns]*rows)
    x,y = w,w
    for i in range(rows):
        for j in range(columns):
            c = (x-i)**2+(y-j)**2
            radius_2d[i,j] = np.sqrt(c)

    radius_2d_arc = radius_2d*pix_size_arcsec.value
    radius_2d_pc = sm(168,radius_2d_arc)*u.pc

    ### GATHERING DATA IN DICTONARY LISTS

    data = {'jy/beam': data_hr[0,0,...],'jy/pixel': data_jy_per_pixel,'jy/arc2': data_jy_per_arcsec2}

    info = {'beam':beam, 'pix_size':pix_size, 'pix/beam': pixels_per_beam,'pix_size_arcsec': pix_size_arcsec, 'median_jy/beam': median, 
    		'median_jy/arc2': median_jy_per_arc2, 'theta': theta, 'bmaj': bmaj, 'bmin':bmin, 'header':header_hr,'position': position}

    rp = {'jy/arc2':rp}
    rp_radius = {'arc':centers['arc'],'pc':centers['pc']}
    radius_2d= {'arc':radius_2d_arc,'pc':radius_2d_pc}


    return data, info, rp, rp_radius, radius_2d

def density_model(Sarr, Rarr, pr = False):
    '''
    turns intensity into density. 
    lmda = wavelength
    Sarr = intensity array profile
    Rarr = radii
    p = print. do you want to print Sarr kpa and B before printing?
    '''
    T = temp(Rarr,pr)

    B = BlackBody(temperature = T)
    kpa = kappa(lmda)

    B=B(lmda_AA)
    arcsec2_per_sr=4.25e10*u.arcsec**2/u.sr
    ergcmshz_per_jy = 1.0e-23*u.erg/u.cm**2/u.s/u.Hz/u.Jy

    B_jy_arcsec2 = B/arcsec2_per_sr/ergcmshz_per_jy
    

    sigma= Sarr/ (kpa*B_jy_arcsec2)

    if pr==True:
        print("Sarr :", Sarr)
        print("kpa :",kpa)
        print("B: ", B_jy_arcsec2)

    return sigma

def kappa(lmda):
    k = k160*(lmda/(160*u.um))**(-1*beta)
    return k


def temp(Rarr, pr=False):
    Tin = 1300.0*u.K
    Rin = 3* rhr
    Rarr_shape = np.shape(Rarr)
    t_cap = 3000*u.K

    if len(Rarr_shape) == 1: #1D
        T = [0.0*u.K]*len(Rarr)
        for i in range(len(Rarr)):
            #T[i] = Tin*((Rarr[i]/Rin)**(-1/2))
            T[i] = np.minimum(t_cap ,Tin*((Rarr[i]/Rin)**(-1/2)))
    elif len(Rarr_shape)==2: #2D len(Rarr_shape==2)
        row, col = Rarr_shape
        T = np.array([[0.0]*col]*row)*u.K
        for i in range(row):
            for j in range(col):
                T[i][j] = np.minimum(t_cap ,Tin*((Rarr[i][j]/Rin)**(-1/2)))
    else: #single value
        T = np.minimum(t_cap ,Tin*((Rarr/Rin)**(-1/2)))
        if pr==True:
            print("T :", T)

    return T


def convolve_resamp(data_lr,data_hr,info_lr,info_hr):
    '''
    convolves and resamples
    '''

    bmaj_pix = info_lr['bmaj']/(info_hr['pix_size'][0].value)
    bmin_pix = info_lr['bmin']/(info_hr['pix_size'][0].value)

    bmaj_sigma = bmaj_pix/(2*np.sqrt(2*np.log(2)))
    bmin_sigma = bmin_pix/(2*np.sqrt(2*np.log(2)))

    gaussian_2D_kernel = Gaussian2DKernel(x_stddev=bmaj_sigma, y_stddev=bmin_sigma, theta = info_lr['theta'], mode = 'oversample')

    convolved_hr_pix = convolve(data_hr['jy/pixel'][0,0], gaussian_2D_kernel) 

    beam_solid_angle_hr = np.pi * info_hr['beam'][0] * info_hr['beam'][1] / (4*np.log(2))
    beam_solid_angle_lr = np.pi * info_lr['beam'][0] * info_lr['beam'][1] / (4*np.log(2))

    convolved_hr_jy_arc2 = convolved_hr_pix / beam_solid_angle_hr.to(u.arcsec**2).value

    wcs_hr = WCS(info_hr['header'])
    wcs_lr = WCS(info_lr['header'])

    reproj_hr_jy_arc, footprint = reproject_exact((convolved_hr_jy_arc2[np.newaxis,np.newaxis], wcs_hr.celestial), wcs_lr.celestial)

    reproj_hr_jy_arc_norm=reproj_hr_jy_arc*np.nanmax(data_lr['jy/arc2'])/np.nanmax(reproj_hr_jy_arc) #normalizing

    reproj_hr_pixel_norm =(reproj_hr_jy_arc_norm *beam_solid_angle_lr.to(u.arcsec**2).value)

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

    plt.plot(data_lr_centered, label = 'lr', c = lowcolor)
    plt.plot(reproj_hr_pixel_centered, label = 'hrconv', c='maroon')
    plt.plot(csm, 'o-', label='CSM', zorder=9,c='purple')
    plt.legend()

    plt.xlim(0,1.5)
    plt.semilogy()
    plt.show()

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
    
    data_conv = {'jy/arc2': reproj_hr_jy_arc_norm, 'jy/pixel': reproj_hr_pixel_norm}
    data_conv_not_norm = {'jy/arc2': reproj_hr_jy_arc}
    data_centered = {'lr': data_lr_centered,'convhr':reproj_hr_pixel_centered}
    data_csm = {'jy/arc2':csm_jy_arc, 'jy/pix': csm}

    info_conv = {'position': position_conv, 'median': median_cv}


    return data_conv, data_conv_not_norm, data_centered, data_csm, info_conv
    
    