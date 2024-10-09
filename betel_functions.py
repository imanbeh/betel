#
# One nice big python file with all of our functions!
#

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
import cmasher as cmr
from astropy.io import fits
from astropy import units as u
from astropy.visualization import quantity_support
quantity_support()  

from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus

r_betelgeuse = 29.50*1e-3*u.arcsec


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




