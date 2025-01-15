'''
density modeler!
'''


import numpy as np
from astropy.modeling.models import BlackBody
from astropy import units as u
from astropy.visualization import quantity_support
quantity_support()  

r_betelgeuse = 29.50*1e-3*u.arcsec
beta= 1.4
k160 = 8.8*u.cm**2 *u.g**-1
stellar_radius_hr_pc = (0.035*168/206265)*u.parsec
rhr = stellar_radius_hr_pc #pc hr radius
lmda = 887.0*u.um #microns
lmda_cm = 0.0887*u.cm
lmda_AA = 8.87e6*u.AA


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