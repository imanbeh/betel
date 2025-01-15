from astropy.io import fits
from astropy import units as u
import numpy as np

from matplotlib import pyplot as plt
import cmasher as cmr




# saving PM CORRECTED low res file header and data info as variable
hdu_lr = fits.open("lr_pm.fits")
data_lr_jy_beam = hdu_lr[0].data
header_lr = hdu_lr[0].header

# saving high res file header and data info as variable
hdu_hr = fits.open("hr_member.uid___A001_X2de_Xf7.ari_l.Betelgeuse_sci.spw0_1_2_3_4_338086MHz.12m.cont.I.pbcor.fits")
data_hr_jy_beam = hdu_hr[0].data
header_hr = hdu_hr[0].header

r_betelgeuse = 29.50*1e-3*u.arcsec

lowcolor = 'steelblue'
highcolor='crimson'

# calculating beam pixel conversions
beam_hr = (header_hr['BMAJ'] * u.deg, header_hr['BMIN']*u.deg)
pix_size_hr = (np.abs(header_hr['CDELT1'])*u.deg,np.abs(header_hr['CDELT2'])*u.deg)
pix_size_arcsec_hr = pix_size_hr[0].to(u.arcsec)
beam_solid_angle_hr =np.pi * beam_hr[0] * beam_hr[1] / (4*np.log(2))

beam_lr = (header_lr['BMAJ'] * u.deg, header_lr['BMIN']*u.deg)
pix_size_lr = (np.abs(header_lr['CDELT1'])*u.deg,np.abs(header_lr['CDELT2'])*u.deg)
pix_size_arcsec_lr = pix_size_lr[0].to(u.arcsec)
beam_solid_angle_lr = np.pi * beam_lr[0] * beam_lr[1] / (4*np.log(2))

# dividing by arcsec conversion to get from beam to arcsec
beam_solid_angle_arcsec2_lr = beam_solid_angle_lr.to(u.arcsec**2).value
pixels_per_beam_lr = beam_solid_angle_lr / (pix_size_lr[0]*pix_size_lr[1])

beam_solid_angle_arcsec2_hr = beam_solid_angle_hr.to(u.arcsec**2).value # dividing by arcsec conversion to get from beam to arcsec
pixels_per_beam_hr = beam_solid_angle_hr / (pix_size_hr[0]*pix_size_hr[1])

data_jy_per_pixel_hr = data_hr_jy_beam / pixels_per_beam_hr
data_jy_per_pixel_lr = data_lr_jy_beam / pixels_per_beam_lr

data_jy_per_arcsec2_hr = data_jy_per_pixel_hr / beam_solid_angle_arcsec2_hr
data_jy_per_arcsec2_lr = data_jy_per_pixel_lr / beam_solid_angle_arcsec2_lr

datas = [data_hr_jy_beam,data_jy_per_pixel_hr,data_jy_per_arcsec2_hr,data_lr_jy_beam,data_jy_per_pixel_lr,data_jy_per_arcsec2_lr]
titles= ['hr jy beam', 'hr jy pixel','hr jy arc2','lr jy beam', 'lr jy pixel',  'lr jy arc2']

# finding max val for each dataset
for i in range(len(datas)):
    max_pos = np.unravel_index(np.argmax(np.ma.masked_invalid(datas[i])), datas[i].shape)
    #print(max_pos)
    y = max_pos[-1]
    x = max_pos[-2]
position_hr=(1273,1290)
position_lr=(439,442)

data_hr = {"jy_pix": data_jy_per_pixel_hr, "jy_arc2": data_jy_per_arcsec2_hr, "jy_beam": data_hr_jy_beam}
data_lr = {"jy_pix": data_jy_per_pixel_lr, "jy_arc2": data_jy_per_arcsec2_lr, "jy_beam": data_lr_jy_beam}


info_hr = {'beam': beam_hr,'beam_solid_angle': beam_solid_angle_hr, 'pix_size': pix_size_hr, 'pix/beam': pixels_per_beam_hr,'pix_size_arcsec': pix_size_arcsec_hr, 
           'theta': header_hr['BPA'], 'bmaj': header_hr['BMAJ'], 'bmin':header_hr['BMIN'], 'header':header_hr,'position': position_hr}

info_lr = {'beam': beam_lr, 'beam_solid_angle': beam_solid_angle_lr,'pix_size': pix_size_lr, 'pix/beam': pixels_per_beam_lr,'pix_size_arcsec': pix_size_arcsec_lr, 
           'theta': header_lr['BPA'], 'bmaj': header_lr['BMAJ'], 'bmin':header_lr['BMIN'], 'header':header_lr,'position': position_lr}

# import pickle

# with open('info_hr.pickle', 'wb') as f:
#     pickle.dump(info_hr, f)

# with open('keys.pickle', 'wb') as f:
#     pickle.dump(info_lr, f)

# with open('keys.pickle', 'wb') as f:
#     pickle.dump(data_hr, f)

# with open('keys.pickle', 'wb') as f:
#     pickle.dump(data_lr, f)