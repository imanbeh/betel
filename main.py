from betel_functions import rp_annulus, radial_read,radius2d,sm
from conv2 import convolve_resamp
from density import density_model
from matplotlib import pyplot as plt
import numpy as np
from astropy import units as u

import sys
import os


# data read in
low_file = "member.uid___A001_X2fb_X200.ari_l.Betelgeuse_sci.spw0_1_2_3_4_338083MHz.12m.cont.I.pbcor.fits"
high_file = "hr_member.uid___A001_X2de_Xf7.ari_l.Betelgeuse_sci.spw0_1_2_3_4_338086MHz.12m.cont.I.pbcor.fits"


data_hr,info_hr,rp_hr,hr_rp_radius,hr_radius_2d = radial_read(high_file, 100)
data_lr,info_lr,rp_lr,lr_rp_radius,lr_radius_2d = radial_read(low_file, 100,x2=442-0.73388889, y2=439+0.26666667)

print(info_lr['position'])

num_1d_model_hr = density_model(rp_hr['jy/arc2'].value,(hr_rp_radius['pc']))
num_1d_model_lr = density_model(rp_lr['jy/arc2'].value,(lr_rp_radius['pc']))

print(data_lr['jy/beam'].shape)
plt.imshow(data_lr['jy/beam'])
err=5
plt.plot(info_lr['position'][0],info_lr['position'][1],'rx')
plt.plot(443.16,439.41,'bx')
plt.xlim(info_lr['position'][0]-err,info_lr['position'][0]+err)
plt.ylim(info_lr['position'][1]-err,info_lr['position'][1]+err)
plt.show()

lowcolor = 'steelblue'
highcolor='crimson'


# plt.style.use("default")
# plt.rcParams["font.family"] = "times"
# plt.rcParams.update({'font.size': 16})
# text_size = 20

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize = [14,6])
# fig.suptitle('Betelgeuse 1D Low Resolution Intensity and Density Profiles',size = 20,y=1)

# ax1.errorbar(hr_rp_radius['arc'],rp_hr['jy/arc2'].value,yerr= np.abs(info_hr['median_jy/arc2'].value),color=highcolor,label = 'High Res',zorder = 0, alpha=0.5,fmt='-o')
# ax1.errorbar(lr_rp_radius['arc'],rp_lr['jy/arc2'].value,yerr = np.abs(info_lr['median_jy/arc2'].value),color=lowcolor,label = 'Low Res',zorder=0, alpha=0.5,fmt= 'o-')
# ax1.semilogy()
# ax1.set_xlim(0, 1.5)
# ax1.set_ylim(1e-2,1e2)

# ax1.set_title("Radial Intensity Profile",size=text_size)
# ax1.set_xlabel("Radius (arcsec)",size=text_size)
# ax1.set_ylabel("Intensity (Jy/arc2)",size=text_size)

# ax2.set_title("Radial Density Profile",size=text_size)
# ax2.errorbar(lr_rp_radius['arc'],num_1d_model_lr.value, yerr = np.abs(info_hr['median_jy/arc2'].value), color = lowcolor,zorder = 0, alpha=0.5,fmt='-o')
# ax2.errorbar(hr_rp_radius['arc'],num_1d_model_hr.value, yerr = np.abs(info_lr['median_jy/arc2'].value), color = highcolor,zorder=0, alpha=0.5,fmt= 'o-')
# ax2.set_xlabel("Radius (arcsec)",size=text_size)
# ax2.set_ylabel("Density (g/cm2)",size=text_size)
# ax2.semilogy()
# ax2.set_xlim(0, 1.5)
# ax2.set_ylim(1e-4,1)
# plt.show()


data_conv_norm, data_conv, data_centered, data_csm, info_conv = convolve_resamp(data_lr, data_hr, info_lr, info_hr)

rp_lr_centered = rp_annulus(data_centered['lr'], info_lr['pix_size_arcsec'], (200,200))
rp_conv_resampled_centered = rp_annulus(data_centered['convhr'], info_lr['pix_size_arcsec'], (200,200))
rp_csm_pix= rp_annulus(data_csm['jy/pix'], info_lr['pix_size_arcsec'],(200,200))
rp_conv_not_norm = rp_annulus(data_conv['jy/arc2'][0,0,...],info_lr['pix_size_arcsec'],info_conv['position'])

plt.plot(rp_lr_centered[0]['arc'],rp_lr_centered[1].value, label = 'lr', c = lowcolor)
plt.plot(rp_conv_resampled_centered[0]['arc'],rp_conv_resampled_centered[1].value, label = 'hrconv', c='maroon')
plt.plot(rp_csm_pix[0]['arc'], rp_csm_pix[1].value, 'o-', label='CSM', zorder=9,c='purple')
plt.legend()
plt.xlim(0,1.5)
plt.semilogy()
plt.show()

print(type(data_csm['jy/arc2']))

from astropy.io import fits

hdu = fits.PrimaryHDU(data_conv['jy/pixel'][0,0,...])
hdu.writeto('data_conv_not_norm.fits',overwrite=True)

hduhr = fits.PrimaryHDU(data_hr['jy/pixel'][0,0,...].value)
hduhr.writeto('data_hr.fits',overwrite=True)



hr_rms_arc = np.nanstd(data_hr['jy/arc2'][0,0][data_hr['jy/arc2'][0,0]<1])
lr_rms_arc = np.nanstd(data_lr['jy/arc2'][0,0][data_lr['jy/arc2'][0,0]<1])
csm_rms_arc = np.nanstd(data_csm['jy/arc2'][data_csm['jy/arc2']<1])
hr_conv_resampled_rms_arc = np.nanstd(data_conv_norm['jy/arc2'][0,0,...][data_conv_norm['jy/arc2'][0,0,...]<0.1])

rp_lr = rp_annulus(data_lr['jy/arc2'][0,0,...], info_lr['pix_size_arcsec'], info_lr['position'])
rp_hr = rp_annulus(data_hr['jy/arc2'][0,0,...], info_hr['pix_size_arcsec'], info_hr['position'])
rp_csm = rp_annulus(data_csm['jy/arc2'], info_lr['pix_size_arcsec'],(200,200))
rp_conv_resampled = rp_annulus(data_conv_norm['jy/arc2'][0,0,...],info_lr['pix_size_arcsec'],info_conv['position'])

hr_1d_density = density_model(rp_hr[1],rp_hr[0]['pc'])
lr_1d_density = density_model(rp_lr[1],rp_lr[0]['pc'])
csm_1d_density = density_model(rp_csm[1],rp_csm[0]['pc'])
hr_conv_resampled_1d_density = density_model(rp_conv_resampled[1],rp_conv_resampled[0]['pc'])

## plotting 1d results

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = [14,5])

ax1.set_title("Radial Intensity in 1D")
ax1.errorbar(rp_hr[0]['arc'], rp_hr[1].value, yerr = hr_rms_arc.value, #np.sqrt(np.sqrt(rp_hr[1].value)**2 + hr_rms**2), 
             fmt='o-', label='HR', zorder=0, alpha=0.5,c=highcolor)

ax1.plot(rp_lr[0]['arc'], rp_lr[1], 'o-', label='LR', zorder=10,c=lowcolor)
ax1.plot(rp_csm[0]['arc'], rp_csm[1], 'o-', label='CSM', zorder=9,c='purple')
ax1.errorbar(rp_conv_resampled[0]['arc'], rp_conv_resampled[1].value, yerr=hr_conv_resampled_rms_arc.value, #np.sqrt(rp_conv_resampled[1].value) + hr_conv_resampled_rms,
               fmt='o-', label='RC HR', zorder=8, alpha=0.75,c='maroon')

ax1.set_xlabel("Aperture size (arcsec)")
ax1.set_ylabel("Surface brightness (Jy arcsec^-2)")
ax1.legend(loc = 'upper right')
ax1.set_xlim(0, 1.5)
ax1.semilogy()
# ax1.ylim(1e-1, 1e7)
#ax1.ylim(-50,1e2)
 

ax2.set_title("Density Analysis 1D")
ax2.errorbar(rp_hr[0]['arc'], hr_1d_density.value, yerr = hr_rms_arc, #np.sqrt(np.sqrt(rp_hr[1].value)**2 + hr_rms**2), 
             fmt='o-', label='HR', zorder=0, alpha=0.5, c = highcolor)

ax2.errorbar(rp_lr[0]['arc'], lr_1d_density.value,yerr = lr_rms_arc, fmt = 'o-', label='LR', zorder=10, c = lowcolor)#yerr = lr_rms_arc
ax2.errorbar(rp_csm[0]['arc'],csm_1d_density.value, yerr=csm_rms_arc,fmt='o-', label='CSM', zorder=9, c = 'purple')
ax2.errorbar(rp_conv_resampled[0]['arc'], hr_conv_resampled_1d_density.value, yerr=hr_conv_resampled_rms_arc.value, #np.sqrt(rp_conv_resampled[1].value) + hr_conv_resampled_rms,
               fmt='o-', label='Resampled Convolved HR', zorder=8, alpha=0.75,color = 'maroon')

ax2.set_xlabel("Aperture size (arcsec)")
ax2.set_ylabel("Density (g cm^-3)")
ax2.legend(loc = 'upper right')
ax2.semilogy()
ax2.set_xlim(0, 1.5)
ax2.set_ylim(1e-5, 1e4)# 
# ax2.xlim(0, 4)
# ax2.ylim(-8.51,8)
plt.show()



radius_csm_2d_arc = radius2d(data_csm['jy/arc2'])*info_lr['pix_size_arcsec']#*u.arcsec
radius_csm_2d_pc = sm(168,radius_csm_2d_arc.value)*u.pc

lr_2d_density = density_model(data_lr['jy/arc2'][0,0,...],lr_radius_2d['pc'])
hr_2d_density = density_model(data_hr['jy/arc2'][0,0,...],hr_radius_2d['pc'])
hr_conv_2d_density = density_model(data_conv_norm['jy/arc2'],hr_radius_2d['pc'])
csm_2d_density = density_model(data_csm['jy/arc2'],radius_csm_2d_pc)

hr_dens_rms_arc_2d = np.nanstd(hr_2d_density.value[hr_2d_density.value<1])
lr_dens_rms_arc_2d = np.nanstd(lr_2d_density.value[lr_2d_density.value<1])
#hr_dens_conv_rms_arc = np.nanstd(result_jy_arc[result_jy_arc<1.0])
hr_dens_conv__rms_arc_2d = np.nanstd(hr_conv_2d_density.value[hr_conv_2d_density.value<0.1])
csm_dens_rms_arc_2d = np.nanstd(csm_2d_density.value[csm_2d_density.value<1])

plt.style.use("default")
plt.rcParams["font.family"] = "times"
plt.rcParams.update({'font.size': 16})
text_size = 15

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 6))  # Adjusted figsize

# Unpacking axes
ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes.flatten()

#fig.suptitle("Intensity Analysis of Betelgeuse in Jy/arc2", size=25, y=1.15)
minn = -0.4
maxx = 0.4
ax1.set_xlim(minn,maxx)
ax1.set_ylim(minn,maxx)
ax2.set_xlim(minn,maxx)
ax2.set_ylim(minn,maxx)
ax3.set_xlim(minn,maxx)
ax4.set_xlim(minn,maxx)
ax3.set_ylim(minn,maxx)
ax4.set_ylim(minn,maxx)
ax5.set_xlim(minn,maxx)
ax5.set_ylim(minn,maxx)
ax6.set_xlim(minn,maxx)
ax6.set_ylim(minn,maxx)
ax7.set_xlim(minn,maxx)
ax7.set_ylim(minn,maxx)
ax8.set_xlim(minn,maxx)
ax8.set_ylim(minn,maxx)

xaxis1 = range(data_hr['jy/arc2'][0,0,...].shape[1])*info_hr['pix_size_arcsec']-((info_hr['position'][0])*info_hr['pix_size_arcsec'])
yaxis1 = range(data_hr['jy/arc2'][0,0,...].shape[0])*info_hr['pix_size_arcsec']-((info_hr['position'][1])*info_hr['pix_size_arcsec'])
ax1.plot(info_hr['position'][0],info_hr['position'][1], c='r')
im1 = ax1.pcolormesh(xaxis1,yaxis1,data_hr['jy/arc2'][0,0,...],cmap = cmr.flamingo,shading = "nearest")
fig.colorbar(im1,ax=ax1)
ax1.set_title("HR Data",size=text_size, y = 1.05)
ax1.set_xlabel("Radius (arc)",size=text_size)
ax1.set_ylabel("Radius (arc)",size=text_size)
#ax1.plot(position_hr[1], position_hr[0], 'rx')
#ax1.contour(xaxis1,yaxis1,data_hr['jy/arc2'][0,0,...].value, colors = 'thistle')

xaxis2 = range(data_lr['jy/arc2'][0,0,...].shape[1])*info_lr['pix_size_arcsec']-(info_lr['position'][0])*info_lr['pix_size_arcsec']
yaxis2 = range(data_lr['jy/arc2'][0,0,...].shape[0])*info_lr['pix_size_arcsec']-(info_lr['position'][1])*info_lr['pix_size_arcsec']
im2 = ax2.pcolormesh(xaxis2,yaxis2,data_lr['jy/arc2'][0,0,...],cmap = cmr.flamingo,shading = "nearest")
fig.colorbar(im2,ax=ax2)
ax2.set_title("LR Data",size=text_size, y = 1.05)
ax2.set_xlabel("Radius (arc)",size=text_size)
ax2.set_ylabel("Radius (arc)",size=text_size)


xaxis3 = range(data_conv_norm['jy/arc2'].shape[1])*info_lr['pix_size_arcsec']-(position_conv[0])*info_lr['pix_size_arcsec']
yaxis3 = range(data_conv_norm['jy/arc2'].shape[0])*info_lr['pix_size_arcsec']-(position_conv[1])*info_lr['pix_size_arcsec']

#im4 = ax4.pcolormesh(xaxis4,yaxis4,reproj_hr_jy_arc[0,0,...]*normalization_arc,cmap = cmr.flamingo,shading = "nearest")#,norm=LogNorm(vmin=1e-5, vmax=1e-3))
im3 = ax3.pcolormesh(xaxis3,yaxis3,data_conv_norm['jy/arc2'],cmap = cmr.flamingo,shading = "nearest")#,norm=LogNorm(vmin=1e-5, vmax=1e-3))
fig.colorbar(im3,ax=ax3)

ax3.set_title("Conv After Reprojection and Normalization",size=text_size, y = 1.05)
ax3.set_xlabel("Radius (arc)",size=text_size)
ax3.set_ylabel("Radius (arc)",size=text_size)



xaxis4 = range(data_csm['jy/arc2'].shape[1])*info_lr['pix_size_arcsec']-(csm.shape[1]/2)*info_lr['pix_size_arcsec']
yaxis4 = range(data_csm['jy/arc2'].shape[0])*info_lr['pix_size_arcsec']-(csm.shape[0]/2)*info_lr['pix_size_arcsec']

im4 = ax4.pcolormesh(xaxis4,yaxis4,data_csm['jy/arc2'],cmap = cmr.flamingo,shading = "nearest")
fig.colorbar(im4,ax=ax4)
ax4.plot(0,0,'rx')

ax4.set_title("Circumstellar Material",size=text_size, y = 1.05)
ax4.set_xlabel("Radius (arc)",size=text_size)
ax4.set_ylabel("Radius (arc)",size=text_size)


xaxis5 = range(data_hr['jy/arc2'][0,0,...].shape[1])*info_hr['pix_size_arcsec']-((info_hr['position'][0])*info_hr['pix_size_arcsec'])
yaxis5 = range(data_hr['jy/arc2'][0,0,...].shape[0])*info_hr['pix_size_arcsec']-((info_hr['position'][1])*info_hr['pix_size_arcsec'])
ax5.plot(info_hr['position'][0],info_hr['position'][1], c='r')
im5 = ax5.pcolormesh(xaxis5,yaxis5,hr_2d_density.value,cmap = cmr.flamingo,shading = "nearest")
fig.colorbar(im5,ax=ax5)
ax5.set_title("HR Data",size=text_size, y = 1.05)
ax5.set_xlabel("Radius (arc)",size=text_size)
ax5.set_ylabel("Radius (arc)",size=text_size)
#ax1.plot(position_hr[1], position_hr[0], 'rx')
#ax1.contour(xaxis1,yaxis1,data_hr['jy/arc2'][0,0,...].value, colors = 'thistle')

xaxis6 = range(data_lr['jy/arc2'][0,0,...].shape[1])*info_lr['pix_size_arcsec']-(info_lr['position'][0])*info_lr['pix_size_arcsec']
yaxis6 = range(data_lr['jy/arc2'][0,0,...].shape[0])*info_lr['pix_size_arcsec']-(info_lr['position'][1])*info_lr['pix_size_arcsec']
im6 = ax6.pcolormesh(xaxis6,yaxis6,lr_2d_density.value,cmap = cmr.flamingo,shading = "nearest")
fig.colorbar(im6,ax=ax6)
ax6.set_title("LR Data",size=text_size, y = 1.05)
ax6.set_xlabel("Radius (arc)",size=text_size)
ax6.set_ylabel("Radius (arc)",size=text_size)



xaxis7 = range(hr_conv_2d_density.shape[1])*info_lr['pix_size_arcsec']-(position_conv[0])*info_lr['pix_size_arcsec']
yaxis7 = range(hr_conv_2d_density.shape[0])*info_lr['pix_size_arcsec']-(position_conv[1])*info_lr['pix_size_arcsec']

#im4 = ax4.pcolormesh(xaxis4,yaxis4,reproj_hr_jy_arc[0,0,...]*normalization_arc,cmap = cmr.flamingo,shading = "nearest")#,norm=LogNorm(vmin=1e-5, vmax=1e-3))
im7 = ax7.pcolormesh(xaxis7,yaxis7,hr_conv_2d_density.value,cmap = cmr.flamingo,shading = "nearest")#,norm=LogNorm(vmin=1e-5, vmax=1e-3))
fig.colorbar(im7,ax=ax7)

ax7.set_title("Conv After Reprojection and Normalization",size=text_size, y = 1.05)
ax7.set_xlabel("Radius (arc)",size=text_size)
ax7.set_ylabel("Radius (arc)",size=text_size)


xaxis8 = range(csm_2d_density.shape[1])*info_lr['pix_size_arcsec']-(data_csm['jy/arc2'].shape[1]/2)*info_lr['pix_size_arcsec']
yaxis8 = range(csm_2d_density.shape[0])*info_lr['pix_size_arcsec']-(data_csm['jy/arc2'].shape[0]/2)*info_lr['pix_size_arcsec']

im8 = ax8.pcolormesh(xaxis8,yaxis8,csm_2d_density.value,cmap = cmr.flamingo,shading = "nearest")
fig.colorbar(im8,ax=ax8)

ax8.set_title("Circumstellar Material",size=text_size, y = 1.05)
ax8.set_xlabel("Radius (arc)",size=text_size)
ax8.set_ylabel("Radius (arc)",size=text_size)

plt.subplots_adjust(hspace=0.5, wspace=0.5)

# ax1.set_xlim(10,20)
# ax1.set_ylim(10,20)





