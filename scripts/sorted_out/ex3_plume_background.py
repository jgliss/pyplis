# -*- coding: utf-8 -*-
"""
piSCOPE example script 3: Plume background analysis

@author: jg
"""
import numpy as np
from os.path import join
import piscope
import matplotlib.pyplot as plt

plt.close("all")
### Define paths of example plume and background image
# Image base path
imgDir = "../data/piscope_etna_testdata/images/"

plumeFile = join(imgDir, 'EC2_1106307_1R02_2015091607065477_F01_Etnaxxxxxxxxxxxx.fts')
bgFile = join(imgDir, 'EC2_1106307_1R02_2015091607022602_F01_Etnaxxxxxxxxxxxx.fts')
offsetFile = join(imgDir, "EC2_1106307_1R02_2015091607064723_D0L_Etnaxxxxxxxxxxxx.fts")
darkFile = join(imgDir, "EC2_1106307_1R02_2015091607064865_D1L_Etnaxxxxxxxxxxxx.fts")

### Specify whether you would like to save the created figure
saveDir = ""
saveFigs = False

### Set options for background retrieval using poly surface fit
maskThresh = 3550 #threshold to init mask for bg surface fit
vignCorr = 1
polyOrder = 1
pyrLevel = 4

### Some options for plotting
tauMin, tauMax = -0.30, 1.8
hspace,wspace = 0.02, 0.02
hPl=.3
w, h = 40, 40

### Create background modellin object
m = piscope.PlumeBackground.PlumeBackgroundModel()

### Define default gas free areas in plume image
#
scale   =   [1200, 80 , 1200 + w, 80 + h]
xgrad   =   [760 , 80 , 760 + w , 80 + h]
ygrad   =   [1200, 520, 1200 + w, 520 + h]

### Define exemplary plume cross section line
pcs     =   [420, 655, 860, 200]
### Define coordinates of horizontal and vertical profile lines 
horLineY = 100
vertLineX = 1220

### Load the image objects and peform dark correction
plume, bg = piscope.Img(plumeFile), piscope.Img(bgFile)
dark, offset = piscope.Img(darkFile), piscope.Img(offsetFile)

horYLabels = [0.0,.10,.20]
vertXLabels = [.0,.7,1.4]

h0,w0 = bg.img.shape
# Model dark image for tExp of plume image
darkPlume = piscope.Processing.model_dark_image(plume, dark, offset)
# Model dark image for tExp of background image
darkBg = piscope.Processing.model_dark_image(bg, dark, offset)

plume.subtract_dark_image(darkPlume)
bg.subtract_dark_image(darkBg)
### Blur the images (sigma = 1)
plume.add_gaussian_blurring(1)
bg.add_gaussian_blurring(1)

### Create vignetting correction mask from background image 
vign = bg.img / bg.img.max() #NOTE: potentially includes y and x gradients
plumeCorr = plume.img / vign

### Create tau0 image (i.e. log(bg/plume))
tau0 = np.log(bg.img/plume.img) #numpy array

### First method: scale background image to plume image in "scale" rect
tau1 = m.get_tau_image(plume, bg, corrMode = 1, scaleRect = scale).img
### Second method: Linear correction for radiance differences based on two rectangles (scale, ygrad)
tau2 = m.get_tau_image(plume, bg, corrMode = 2, scaleRect = scale,\
    yGradRect = ygrad).img
### Third method: 2nd order polynomial fit along vertical profile line
### For this method, determine tau on tau off and AA image
tau3 = m.get_tau_image(plume, bg, corrMode = 3, yGradLineX = vertLineX,\
    yGradLineStart = 10, yGradLineStop = 590, yGradLinePolyOrder = 2).img

### Create mask for poly surface fit  
mask = np.ones(plumeCorr.shape, dtype = np.float32)   
mask[plumeCorr < maskThresh] = 0
### Fourth Method: retrieve tau image using poly surface fit
tau4 = m.get_tau_image(piscope.Img(plumeCorr), corrMode = 0, surfaceFitMask =\
    mask, surfaceFitPolyOrder = 1).img


#==============================================================================
# ### Create :class:`LineOnImage` objects for line profile retrievals 
# lvert = piscope.Processing.LineOnImage(vertLineX, 0, vertLineX, 1023, id = "vert")
# lhor = piscope.LineOnImage(0, horLineY, 1343, horLineY, id = "hor")
# pcsL = piscope.LineOnImage(*pcs, id="pcs")
# 
# 
# p1_0 = lvert.get_line_profile(tau1)
# p1_1 = lvert.get_line_profile(tau2)
# p1_2 = lvert.get_line_profile(tau3)
# p1_3 = lvert.get_line_profile(tau4)
# 
# p2_0 = lhor.get_line_profile(tau1)
# p2_1 = lhor.get_line_profile(tau2)
# p2_2 = lhor.get_line_profile(tau3)
# p2_3 = lhor.get_line_profile(tau4)
#==============================================================================



#==============================================================================
# 
# ### Plot PCS profiles for method 3 (on, off, aa)
# fig0, ax0 = plt.subplots(1,1)
# 
# x = np.arange(0, 633, 1)
# pOn = pcsL.get_line_profile(tau3)
# 
# ax0.plot(pOn,"--", label=r"On: $\phi=%.3f$"%(sum(pOn)/633))
# ax0.plot(pOff,"--", label=r"Off: $\phi=%.3f$"%(sum(pOff)/633))
# ax0.plot(pAA,"-", lw = 2, label=r"AA: $\phi=%.3f$"%(sum(pAA)/633))
# #ax0.fill_between(x, pOn, pOff, facecolor="r", alpha = 0.1)
# 
# ax0.legend(loc=3, fancybox=True, framealpha=0.5, fontsize=12)
# ax0.grid()
# ax0.set_ylabel(r"$\tau$", fontsize=20)
# ax0.set_xlim([0,632])
# ax0.set_xticklabels([])
# ax0.set_xlabel("PCS", fontsize=16)
# #ax1.plot(pcsL.get_line_profile(tau1))
# 
# 
# ### Plot PCS profiles for all 4 methods (Fig. XX in manuscript)
# fig1, ax1 = plt.subplots(1,1)
# p1 = pcsL.get_line_profile(tau1)
# p2 = pcsL.get_line_profile(tau2)
# p3 = pcsL.get_line_profile(tau3)
# p4 = pcsL.get_line_profile(tau4)
# ax1.plot(p1,"-", label=r"A) scaled: $\phi=%.3f$"%(sum(p1)/633))
# ax1.plot(p2,"-", label=r"B) lin. corr: $\phi=%.3f$"%(sum(p2)/633))
# ax1.plot(p3,"-", label=r"C) quad. corr: $\phi=%.3f$"%(sum(p3)/633))
# ax1.plot(p4,"-", label=r"D) 2D surface fit: $\phi=%.3f$"%(sum(p4)/633))
# 
# ax1.legend(loc=3, fancybox=True, framealpha=0.5, fontsize=12)
# ax1.grid()
# ax1.set_ylabel(r"$\tau$", fontsize=20)
# ax1.set_xlim([0,632])
# ax1.set_xticklabels([])
# ax1.set_xlabel("PCS", fontsize=16)
# #ax1.plot(pcsL.get_line_profile(tau1))
# 
# ### Plot result first method
# fig2 = plt.figure()
# gs = plt.GridSpec(2, 2, width_ratios=[w0,w0*hPl], height_ratios=[h0*hPl,h0])
# ax = [plt.subplot(gs[2]),]
# ax.append(plt.subplot(gs[3]))#, sharey=ax[0]))
# ax.append(plt.subplot(gs[0]))#, sharex=ax[0]))
# 
# ax[0].imshow(tau1, cmap = cmap, vmin = tauMin, vmax = tauMax)
# ax[0].plot([vertLineX,vertLineX],[0,1023], "-b", label = "vert profile")
# ax[0].plot([0,1343],[horLineY,horLineY], "-c", label = "hor profile")
# ax[0].plot([pcs[0],pcs[2]],[pcs[1],pcs[3]], "--",c="g", label="PCS")
# #ax[0].set_title("Tau img after scaling", fontsize=14)
# #fig.suptitle("Tau img after scaling", fontsize=14)
# ax[0].add_patch(Rectangle((scale[0], scale[1]), w, h, ec="g",fc="none",\
#                                             label = "scale rect"))
# 
# ax[0].legend(loc="best", fancybox=True, framealpha=0.5, fontsize=10)
# 
# #ax[0].xaxis.tick_top()   
# ax[2].set_xticklabels([])
# ax[1].set_yticklabels([])
# 
# #plot vertical profile
# ax[1].plot(p1_0, np.arange(0, len(p1_0),1), "-b" , label = "vert profile")
# ax[1].set_xlabel(r"$\tau$", fontsize=16)
# ax[2].set_ylabel(r"$\tau$", fontsize=16)
# ax[1].yaxis.tick_right()   
# ax[1].set_ylim([1023,0])
# ax[1].get_xaxis().set_ticks(vertXLabels)
# ax[1].set_xlim([-.2,1.6])
# 
# #plot horizontal profile
# ax[2].plot(np.arange(0, len(p2_0),1),p2_0, "-c" , label = "hor profile")
# ax[2].get_yaxis().set_ticks(horYLabels)
# ax[2].set_ylim([-.05,.25])
# ax[2].set_xlim([0,1343])
# 
# plt.subplots_adjust(wspace=wspace, hspace=hspace)
# 
# ax[2].axhline(0, ls="--", color="k")
# ax[1].axvline(0, ls="--", color="k")
# 
# ax[0].set_xlim([0,1343])
# ax[0].set_ylim([1023,0])
# 
# ### Plot result seond method   
# fig3 = plt.figure()
# gs = plt.GridSpec(2, 2, width_ratios=[w0,w0*.2], height_ratios=[h0,h0*.2])
# gs = plt.GridSpec(2, 2, width_ratios=[w0,w0*hPl], height_ratios=[h0*hPl,h0])
# ax = [plt.subplot(gs[2]),]
# ax.append(plt.subplot(gs[3]))#, sharey=ax[0]))
# ax.append(plt.subplot(gs[0]))#, sharex=ax[0]))
# 
# 
# #plot image    
# ax[0].set_xlim([0,1343])
# ax[0].set_ylim([1023,0])
# 
# ax[0].imshow(tau2, cmap = cmap, vmin = tauMin, vmax = tauMax)
# #ax[0].imshow(tau2, cmap=cmap)
# ax[0].plot([vertLineX,vertLineX],[0,1023], "-b", label = "vert profile")
# ax[0].plot([0,1343],[horLineY,horLineY], "-c", label = "hor profile")
# ax[0].plot([pcs[0],pcs[2]],[pcs[1],pcs[3]], "--",c="g", label="PCS")
# #fig.suptitle("Tau image after scaling + ygrad corr", fontsize=14)
# ax[0].add_patch(Rectangle((scale[0], scale[1]), w, h, ec="g",fc="none",\
#                                             label = "scale rect"))
# ax[0].add_patch(Rectangle((ygrad[0], ygrad[1]), w, h, ec="g",fc="none",\
#                                             label="ygrad rect"))
# 
# ax[0].legend(loc="best", fancybox=True, framealpha=0.5, fontsize=10)
# 
# ax[2].set_xticklabels([])
# ax[1].set_yticklabels([])
# 
# 
# #plot vertical profile
# ax[1].plot(p1_1, np.arange(0, len(p1_1),1), "-b" , label = "vert profile")
# ax[1].yaxis.tick_right()   
# ax[1].set_ylim([1023,0])
# ax[1].get_xaxis().set_ticks(vertXLabels)
# ax[1].set_xlim([-.2,1.6])
# 
# #plot horizontal profile
# ax[2].plot(np.arange(0, len(p2_1),1),p2_1, "-c" , label = "hor profile")
# ax[2].get_yaxis().set_ticks(horYLabels)
# ax[2].set_ylim([-.05,.25])
# ax[2].set_xlim([0,1343])
# 
# plt.subplots_adjust(wspace=wspace, hspace=hspace)
# ax[2].axhline(0, ls="--", color="k")
# ax[1].axvline(0, ls="--", color="k")
# 
# ax[1].set_xlabel(r"$\tau$", fontsize=16)
# ax[2].set_ylabel(r"$\tau$", fontsize=16)
# 
# ### Plot result third method    
# fig4 = plt.figure()
# gs = plt.GridSpec(2, 2, width_ratios=[w0,w0*hPl], height_ratios=[h0*hPl,h0])
# ax = [plt.subplot(gs[2]),]
# ax.append(plt.subplot(gs[3]))#, sharey=ax[0]))
# ax.append(plt.subplot(gs[0]))#, sharex=ax[0]))
# 
# 
# #plot image    
# ax[0].imshow(tau3, cmap = cmap, vmin = tauMin, vmax = tauMax)
# #ax[0].imshow(tau3)
# ax[0].plot([vertLineX,vertLineX],[0,1023], "-b", label = "vert profile")
# ax[0].plot([0,1343],[horLineY,horLineY], "-c", label = "hor profile")
# ax[0].plot([pcs[0],pcs[2]],[pcs[1],pcs[3]], "--",c="g", label="PCS")
# 
# ax[0].set_xlim([0,1343])
# ax[0].set_ylim([1023,0])
# ax[0].legend(loc="best", fancybox=True, framealpha=0.5, fontsize=10)
# 
# ax[2].set_xticklabels([])
# ax[1].set_yticklabels([])
# 
# #plot vertical profile
# ax[1].plot(p1_2, np.arange(0, len(p1_2),1), "-b" , label = "vert profile")
# ax[1].yaxis.tick_right()   
# ax[1].set_ylim([1023,0])
# ax[1].get_xaxis().set_ticks(vertXLabels)
# ax[1].set_xlim([-.2,1.6])
#    
# 
# #plot horizontal profile
# ax[2].plot(np.arange(0, len(p2_2), 1),p2_2, "-c" , label = "hor profile")
# ax[2].get_yaxis().set_ticks(horYLabels)
# ax[2].set_ylim([-.05,.25])
# ax[2].set_xlim([0,1343])
# 
# plt.subplots_adjust(wspace=wspace, hspace=hspace)
# ax[2].axhline(0, ls="--", color="k")
# ax[1].axvline(0, ls="--", color="k")
# 
# ax[1].set_xlabel(r"$\tau$", fontsize=16)
# ax[2].set_ylabel(r"$\tau$", fontsize=16)
# 
# ### Plot result third method (off band) 
# fig4_off = plt.figure()
# gs = plt.GridSpec(2, 2, width_ratios=[w0,w0*hPl], height_ratios=[h0*hPl,h0])
# ax = [plt.subplot(gs[2]),]
# ax.append(plt.subplot(gs[3]))#, sharey=ax[0]))
# ax.append(plt.subplot(gs[0]))#, sharex=ax[0]))
# 
# 
# #plot image    
# ax[0].imshow(tau3Off, cmap = cmap, vmin = tauMin, vmax = tauMax)
# #ax[0].imshow(tau3)
# ax[0].plot([vertLineX,vertLineX],[0,1023], "-b", label = "vert profile")
# ax[0].plot([0,1343],[horLineY,horLineY], "-c", label = "hor profile")
# ax[0].plot([pcs[0],pcs[2]],[pcs[1],pcs[3]], "--",c="g", label="PCS")
# 
# ax[0].set_xlim([0,1343])
# ax[0].set_ylim([1023,0])
# ax[0].legend(loc="best", fancybox=True, framealpha=0.5, fontsize=10)
# 
# ax[2].set_xticklabels([])
# ax[1].set_yticklabels([])
# 
# #plot vertical profile
# ax[1].plot(p1_2Off, np.arange(0, len(p1_2Off),1), "-b" , label = "vert profile")
# ax[1].yaxis.tick_right()   
# ax[1].set_ylim([1023,0])
# ax[1].get_xaxis().set_ticks(vertXLabels)
# ax[1].set_xlim([-.2,1.6])
#    
# 
# #plot horizontal profile
# ax[2].plot(np.arange(0, len(p2_2Off), 1),p2_2Off, "-c" , label = "hor profile")
# ax[2].get_yaxis().set_ticks(horYLabels)
# ax[2].set_ylim([-.05,.25])
# ax[2].set_xlim([0,1343])
# 
# plt.subplots_adjust(wspace=wspace, hspace=hspace)
# ax[2].axhline(0, ls="--", color="k")
# ax[1].axvline(0, ls="--", color="k")
# 
# ax[1].set_xlabel(r"$\tau$", fontsize=16)
# ax[2].set_ylabel(r"$\tau$", fontsize=16)
# 
# ### Plot result third method (off band) 
# fig4_aa = plt.figure()
# gs = plt.GridSpec(2, 2, width_ratios=[w0,w0*hPl], height_ratios=[h0*hPl,h0])
# ax = [plt.subplot(gs[2]),]
# ax.append(plt.subplot(gs[3]))#, sharey=ax[0]))
# ax.append(plt.subplot(gs[0]))#, sharex=ax[0]))
# 
# 
# #plot image    
# ax[0].imshow(aa, cmap = cmap, vmin = tauMin, vmax = tauMax)
# #ax[0].imshow(tau3)
# ax[0].plot([vertLineX,vertLineX],[0,1023], "-b", label = "vert profile")
# ax[0].plot([0,1343],[horLineY,horLineY], "-c", label = "hor profile")
# ax[0].plot([pcs[0],pcs[2]],[pcs[1],pcs[3]], "--",c="g", label="PCS")
# 
# ax[0].set_xlim([0,1343])
# ax[0].set_ylim([1023,0])
# ax[0].legend(loc="best", fancybox=True, framealpha=0.5, fontsize=10)
# 
# ax[2].set_xticklabels([])
# ax[1].set_yticklabels([])
# 
# #plot vertical profile
# ax[1].plot(p1_2aa, np.arange(0, len(p1_2aa),1), "-b" , label = "vert profile")
# ax[1].yaxis.tick_right()   
# ax[1].set_ylim([1023,0])
# ax[1].get_xaxis().set_ticks(vertXLabels)
# ax[1].set_xlim([-.2,1.6])
#    
# 
# #plot horizontal profile
# ax[2].plot(np.arange(0, len(p2_2aa), 1),p2_2aa, "-c" , label = "hor profile")
# ax[2].get_yaxis().set_ticks(horYLabels)
# ax[2].set_ylim([-.05,.25])
# ax[2].set_xlim([0,1343])
# 
# plt.subplots_adjust(wspace=wspace, hspace=hspace)
# ax[2].axhline(0, ls="--", color="k")
# ax[1].axvline(0, ls="--", color="k")
# 
# ax[1].set_xlabel(r"$\tau$", fontsize=16)
# ax[2].set_ylabel(r"$\tau$", fontsize=16)
# 
# ### Plot result fourth method    
# fig5 = plt.figure()
# gs = plt.GridSpec(2, 2, width_ratios=[w0,w0*hPl], height_ratios=[h0*hPl,h0])
# ax = [plt.subplot(gs[2]),]
# ax.append(plt.subplot(gs[3]))#, sharey=ax[0]))
# ax.append(plt.subplot(gs[0]))#, sharex=ax[0]))
# ax.append(plt.subplot(gs[1]))
# 
# palette = colors.ListedColormap(['white', 'lime'])
# norm = colors.BoundaryNorm([0,.5,1], palette.N)
# 
# ax[3].imshow(mask, cmap=palette, norm=norm, alpha = .7)
# ax[3].set_title("Mask", fontsize=10)
# ax[3].set_xticklabels([])
# ax[3].set_yticklabels([])
# 
# ax[0].imshow(tau4, cmap = cmap, vmin = tauMin, vmax = tauMax)
# 
# ax[0].plot([vertLineX,vertLineX],[0,1023], "-b", label = "vert profile")
# ax[0].plot([0,1343],[horLineY,horLineY], "-c", label = "hor profile")
# ax[0].plot([pcs[0],pcs[2]],[pcs[1],pcs[3]], "--",c="g", label="PCS")
# 
# ax[0].set_xlim([0,1343])
# ax[0].set_ylim([1023,0])
# ax[0].legend(loc="best", fancybox=True, framealpha=0.5, fontsize=10)
# 
# ax[2].set_xticklabels([])
# ax[1].set_yticklabels([])
# 
# 
# #plot vertical profile
# ax[1].plot(p1_3, np.arange(0, len(p1_3),1), "-b" , label = "vert profile")
# ax[1].yaxis.tick_right()   
# ax[1].set_ylim([1023,0])
# ax[1].get_xaxis().set_ticks(vertXLabels)
# ax[1].set_xlim([-.2,1.6])
#    
# 
# #plot horizontal profile
# ax[2].plot(np.arange(0, len(p2_3), 1),p2_3, "-c" , label = "hor profile")
# ax[2].get_yaxis().set_ticks(horYLabels)
# ax[2].set_ylim([-.05,.25])
# ax[2].set_xlim([0,1343])
# 
# plt.subplots_adjust(wspace=wspace, hspace=hspace)
# ax[2].axhline(0, ls="--", color="k")
# ax[1].axvline(0, ls="--", color="k")
# 
# ax[1].set_xlabel(r"$\tau$", fontsize=16)
# ax[2].set_ylabel(r"$\tau$", fontsize=16)    
# if saveFigs:
#     fig1.savefig(os.path.join(savePath, "bg_corr_ICAs.png"))
#     fig2.savefig(os.path.join(savePath, "bg_corr_0.png"))
#     fig3.savefig(os.path.join(savePath, "bg_corr_1.png"))
#     fig4.savefig(os.path.join(savePath, "bg_corr_2.png"))
#     fig5.savefig(os.path.join(savePath, "bg_corr_3.png"))
# 
# fig6 = plt.figure()
# plt.imshow(plume.img, cmap = "gray")
#==============================================================================
