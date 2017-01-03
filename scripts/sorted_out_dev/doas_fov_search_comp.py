# -*- coding: utf-8 -*-
"""
This third test imports the SO2-cam data provided by Jonas Gliss on 30 August 2016
and searches for a FOV parametrisation using a 2D Gaussian
and applies the convolution on the camera images
"""

import numpy as np
from piscope.Helpers import shifted_color_map
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsmr
import scipy.optimize as opt
from scipy.ndimage.filters import gaussian_filter
from matplotlib.pyplot import Circle
from pandas import Series
### beginning of function declaration

# function for determining IFR distribution using LSMR regularisation
def IFRlsmr(specDataVector, tauImgStack, lmbda = 1e-6):
    # some input data size checking
    (m,) = specDataVector.shape
    (Ny,Nx,m2) = tauImgStack.shape
    if m != m2:
        raise ValueError('Sizes of specDataVector and tauImgStack not consistent.')
    # construct H-matrix thorugh reshaping image stack
    tauImgStack_ = np.transpose(tauImgStack, (2,0,1))
    H0 = tauImgStack_.reshape(m,Nx*Ny)
    # and one-vector
    H1 = np.ones((m,1), dtype=H0.dtype)
    # and stacking in the end
    H = np.column_stack((H1,H0))
    # solve using LSMR regularisation
    A = lsmr(H, specDataVector, atol=lmbda, btol=lmbda)
    c = A[0]
    # separate offsetand image
    lsmrOffset = c[0]
    lsmrImage = c[1:].reshape(Ny, Nx)
    return [lsmrOffset, lsmrImage]

# define IFR model functions

def gauss2D((x, y), amplitude, xm, ym, xsigma, ysigma, offset):
    U = ((x-xm)/xsigma)**2 + ((y-ym)/ysigma)**2
    g = offset + amplitude * np.exp(-U/2)
    return g.ravel()

def gauss2D_tilt((x, y), amplitude, xm, ym, xsigma, ysigma, offset, theta):
    xprime = (x-xm)*np.cos(theta) - (y-ym)*np.sin(theta)
    yprime = (y-ym)*np.sin(theta) + (y-ym)*np.cos(theta)
    U = (xprime/xsigma)**2 + (yprime/ysigma)**2
    g = offset + amplitude * np.exp(-U/2)
    return g.ravel()

def hypergauss2D((x, y), amplitude, xm, ym, xsigma, ysigma, offset):
    U = ((x-xm)/xsigma)**2 + ((y-ym)/ysigma)**2
    g = offset + amplitude * np.exp(-U**4/2)
    return g.ravel()

def hypergauss2D_tilt((x, y), amplitude, xm, ym, xsigma, ysigma, offset, theta):
    xprime = (x-xm)*np.cos(theta) - (y-ym)*np.sin(theta)
    yprime = (y-ym)*np.sin(theta) + (y-ym)*np.cos(theta)
    U = (xprime/xsigma)**2 + (yprime/ysigma)**2
    g = offset + amplitude * np.exp(-U**4/2)
    return g.ravel()
    
# function running the 2D Gaussian fits    
# Achtung, hyper Gauss und tilt ist signifikant langsamer
    
def FOVgaussfit(lsmrIm, hyper=True, tilt=False, crop=True):
    # setup grid
    (Ny,Nx) = lsmrIm.shape
    xvec = np.linspace(0, Nx, Nx)
    yvec = np.linspace(0, Ny, Ny)
    xgrid, ygrid = np.meshgrid(xvec, yvec)
    # apply maximum of filtered image to initialise 2D gaussian fit
    sigma = 20
### Holger I think x and y need to be swapped (the first returned index in unravel_index is the y coordinate)
    #(x0, y0) = np.unravel_index(np.nanargmax(gaussian_filter(lsmrIm, sigma)), lsmrIm.shape)
    (y0, x0) = np.unravel_index(np.nanargmax(gaussian_filter(lsmrIm, sigma)), lsmrIm.shape)
    if tilt:
        initial_guess = (1, x0, y0, 10, 10, 0, 0)
        if hyper:
            popt, pcov = opt.curve_fit(hypergauss2D_tilt, (xgrid, ygrid), lsmrIm.ravel(), p0=initial_guess)
            data_fitted = hypergauss2D_tilt((xgrid, ygrid), *popt)
        else:
            popt, pcov = opt.curve_fit(gauss2D_tilt, (xgrid, ygrid), lsmrIm.ravel(), p0=initial_guess)
            data_fitted = gauss2D_tilt((xgrid, ygrid), *popt)
    else:
        initial_guess = (1, x0, y0, 10, 10, 0)
        if hyper:
            popt, pcov = opt.curve_fit(hypergauss2D, (xgrid, ygrid), lsmrIm.ravel(), p0=initial_guess)
            data_fitted = hypergauss2D((xgrid, ygrid), *popt)
        else:
            popt, pcov = opt.curve_fit(gauss2D, (xgrid, ygrid), lsmrIm.ravel(), p0=initial_guess)
            data_fitted = gauss2D((xgrid, ygrid), *popt)
    # eventually crop FOV distribution (makes it more robust against outliers (eg. mountan ridge))
    if crop:
        # set outside (one 100th of amplitude) datapoints = 0
        data_fitted[data_fitted<popt[0]/100] = 0
    # reshape data_fitted as matrix instead of vector required for fitting
    data_fitted = data_fitted.reshape(Ny, Nx)
    # normalise fit result
    normsum = np.sum(data_fitted)
    data_fitted_norm = data_fitted/normsum
    popt[0] = popt[0]/normsum
    return [data_fitted_norm, popt]

# function convolving the image stack with the obtained FOV distribution
    
def convolveFOV(tauImgStack, data_fitted_norm):
    tauImgStack_conv = np.transpose(tauImgStack, (2,0,1)) * data_fitted_norm
    calib_curve = tauImgStack_conv.sum((1,2))
    return calib_curve


def estimate_disk_radius(stack, cx, cy, doasData, smooth = 2):
    """Finds disk radius with highest correlation"""
    h, w =  stack.shape[:2]
    #find maximum radius (around CFOV pos) which still fits into the image
    #shape of the stack used to find the best radius
    maxR = min([cx, cy, w - cx, h - cy])
    #radius array
    radii = np.arange(1, maxR, 1)
    y ,x = np.ogrid[:h, :w]
    maxCorr = 0
    radius = None
    tauData = None
    coeffs = []
    for r in radii:
        print r
        mask = ((x - cx)**2 + (y - cy)**2 < r**2).astype(float)
        mask_norm = mask/sum(mask)
        tauSeries = Series(convolveFOV(tauImgStack, mask_norm), doasData.index)
        coeff = tauSeries.corr(doasData)
        coeffs.append(coeff)
        #and append correlation coefficient to results
        if coeff > maxCorr:
            radius = r
            maxCorr = coeff
            tauData = tauSeries
        
        
        
    corrCurve = Series(np.asarray(coeffs, dtype = float), radii)
    return tauData, radius, maxCorr, corrCurve
    

    
    
### end of function declaration

### begining of paramter declaration (maybe get 'em from GUI radio knobs..)

LSMRlambda=1e-6 # default value
G2Dtilt = True
G2Dhyper = True
G2Dcrop = True

### end of parameter declaration

### commands follow
    
# close all figures
plt.close("all")

# import data
datapath = r'D:/Dropbox/Python27/jgliss/modules/piscope/_private/out_in/20160830f_hsihler_fovSearchData/'
#datapath = '/media/hsihler/hsat/nobackup/20160830_hsihler_fovSearchData/'
fn_acqTimes = 'acqTimes'
fn_specDataVector = 'specDataVector'
fn_tauImgStack = 'tauImgStack'

acqTimes = np.load(datapath+fn_acqTimes)
specDataVector = np.load(datapath+fn_specDataVector)
tauImgStack = np.load(datapath+fn_tauImgStack)
### Added stuff to retireve the results based on Pearson correlation

corrIm = np.load(datapath + "corrIm")
so2 = Series(specDataVector, acqTimes) #pandas Series (just convenience)

#get max position in correlation image
(cy, cx) = np.unravel_index(np.nanargmax(gaussian_filter(corrIm, 2)), corrIm.shape)
#define borders
h, w =  tauImgStack.shape[:2]
#eval grid
y ,x = np.ogrid[:h, :w]

tauDataPearson, radius, maxCorr, corrCurve = estimate_disk_radius(tauImgStack, cx, cy, so2, smooth = 2)
#tauValsPearson = convolveFOV(tauImgStack, mask_norm)

# hier gehts eigentlich erst los...

# evaluate data for FOV distribution
(lsmrOf, lsmrIm) = IFRlsmr(specDataVector, tauImgStack, lmbda=LSMRlambda)

# fit FOV distribution using a (tilted) 2D (hyper-)gaussian
(data_fitted_norm, popt) = FOVgaussfit(lsmrIm, hyper=G2Dhyper, tilt=G2Dtilt, crop=G2Dcrop)

# convolve FOV fit-result with image stack to obtain "calibration curve"
tauValsLSMR = convolveFOV(tauImgStack, data_fitted_norm)

# hier ist im Prinzip schon alles wichtige überstanden... ;)

# plot gaussian IFR fit result
(Ny,Nx,m) = tauImgStack.shape
xvec = np.linspace(0, Nx, Nx)
yvec = np.linspace(0, Ny, Ny)
xgrid, ygrid = np.meshgrid(xvec, yvec)

fig, axes = plt.subplots(1, 2, figsize=(18,5))

ax=axes[0]
ax.hold(True)
lsmrDisp = lsmrIm * 100
vmin, vmax = -3.3, 3.3
cmap = shifted_color_map(vmin, vmax, cmap = plt.cm.RdBu)

dispL = ax.imshow(lsmrDisp, vmin=vmin, vmax=vmax, cmap=cmap)#, cmap=plt.cm.jet)
cbL=fig.colorbar(dispL, ax = ax)
cbL.set_label(r"FOV fraction [$10^{-2}$ pixel$^{-1}$]", fontsize=16)
#ax.plot(popt[1], popt[2], 'x')
ax.contour(xgrid, ygrid, data_fitted_norm, (popt[0]/np.e, popt[0]/10), colors='k')
ax.get_xaxis().set_ticks([popt[1]])
ax.get_yaxis().set_ticks([popt[2]])
ax.axhline(popt[2], ls="--", color="k")
ax.axvline(popt[1], ls="--", color="k")
#ax.set_axis_off()
ax.set_title("A) LSMR routine (Parametr: tilted hypergauss)")

ax=axes[1]
ax.hold(True)
#cmap = shifted_color_map(-1, corrIm.max(), cmap = plt.cm.RdBu)
dispR = ax.imshow(corrIm, vmin = -1, vmax = 1, cmap=plt.cm.RdBu)
cbR = fig.colorbar(dispR, ax = ax)
cbR.set_label(r"Pearson corr. coeff.", fontsize = 16)
ax.autoscale(False)
#ax.plot(cx,cy, "x")
c = Circle((cx, cy), radius, ec="k", fc="none")
ax.add_artist(c)
ax.set_title("B) Pearson routine (Parametr: circ. disk)")
ax.get_xaxis().set_ticks([cx])
ax.get_yaxis().set_ticks([cy])
ax.axhline(cy, ls="--", color="k")
ax.axvline(cx, ls="--", color="k")
#ax.set_axis_off()
# plot calibration curve
tauLSMR = Series(tauValsLSMR, acqTimes)
fig.tight_layout()
fig, axes = plt.subplots(1,2, figsize=(16,6))
ax =axes[0]
ts=tauDataPearson.index
p1 = ax.plot(ts, tauDataPearson, "--x", label="tauPearson")
p2 = ax.plot(ts, tauLSMR, "--x", label="tauLSMR")

ax.set_ylabel(r"$\tau$", fontsize=18)
ax2 = ax.twinx()
p3 = ax2.plot(ts, so2,"-r", label = "SO2 CDs")
ax2.set_ylabel(r"SO2 CD [+E17 cm-2]", fontsize=16)
ax.set_title("Time series overlay", fontsize=18)
ps =p1+p2+p3
labs = [l.get_label() for l in ps]        
ax.legend(ps, labs, loc="best",fancybox=True, framealpha=0.5)
plt.draw()

#now compute calibration curves
pl0, V0 = np.polyfit(tauDataPearson, so2, 1, cov=True)
pl1, V1 = np.polyfit(tauLSMR, so2, 1, cov=True)


polyPearson = np.poly1d(pl0)
polyLSMR = np.poly1d(pl1)
x=np.linspace(0, .12)
#print "x_1: {} +/- {}".format(p[0], np.sqrt(V[0][0]))
#print "x_2: {} +/- {}".format(p[1], np.sqrt(V[1][1]))


ax = axes[1]
pd=ax.plot(tauDataPearson, so2, " x", c=p1[0].get_color(), label="pearson data")
lblP="regr: %.2f (+/- %.2f) + %.1f(+/- %.1f)x " %(polyPearson[0], V0[1][1], polyPearson[1], V0[0][0])
ax.plot(x,polyPearson(x), c=p1[0].get_color(), label= lblP)
ld=ax.plot(tauLSMR, so2, " x",c=p2[0].get_color(), label="lsmr data")
lblL="regr: %.2f (+/- %.2f) + %.1f(+/- %.1f)x" %(polyLSMR[0], V1[1][1], polyLSMR[1], V1[0][0])
ax.plot(x,polyLSMR(x), c=p2[0].get_color(), label= lblL)
ax.set_title("Calibration curves", fontsize=18)
ax.set_xlabel(r"$\tau$", fontsize=18)
ax.grid()
ax.legend(loc="best",fancybox=True, framealpha=0.5, fontsize=11).draggable()

