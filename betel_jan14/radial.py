# betelfunctions
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


def radial_read(data,info):
    '''
    reads in file and returns radial profile and returns important data
    enter data in jy/pixel.
    '''

    centers, rp_1d_jy_pix = rp_annulus(data,info['pix_size_arcsec'],info['position'])

    ### RADIUS DATA IN 2D
    r2d = radius2d(data)

    radius_2d_arc = r2d*info['pix_size_arcsec'].value
    radius_2d_pc = sm(168,radius_2d_arc)*u.pc

    # ### GATHERING DATA IN DICTONARY LISTS
    # rp = {'jy/arc2':rp}
    # rp_radius = {'arc':centers['arc'],'pc':centers['pc']}
    # radius_2d= {'arc':radius_2d_arc,'pc':radius_2d_pc}

    radius = {'arc_1d': centers['arc'], 'pc_1d': centers['pc'], 'arc_2d': radius_2d_arc, 'pc_2d': radius_2d_pc}

    return radius, rp_1d_jy_pix


def radius2d(data):
    ### RADIUS DATA IN 2D
    dimensions = data.shape
    rows,columns = dimensions
    radius_2d = np.array([[0.0]*columns]*rows)
    x = rows/2
    y = rows/2
    for i in range(rows):
        for j in range(columns):
            c = (x-i)**2+(y-j)**2
            radius_2d[i,j] = np.sqrt(c)

    return radius_2d

