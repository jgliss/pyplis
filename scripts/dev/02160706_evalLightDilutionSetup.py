# -*- coding: utf-8 -*-
"""
Created on Wed Jul 06 11:22:05 2016

@author: Jonas Gliß
@email: jg@nilu.no
@Copyright: Jonas Gliß
"""
from os import listdir
from os.path import join
from collections import OrderedDict as od
from numpy import loadtxt
from pandas import Series
from scipy.optimize import leastsq
import numpy as np
import matplotlib.pyplot as plt

from mpltools import color


from pygascam.LibradtranStuff import *

p=r'D:/Jonas/Research/Data/Campaigns/201411_Chile/DATA_ANALYSIS/Guallatiri/AnalysisResults/RT/Simulations/spectra/20160706_lightDilution/spectra/'
savep=r'D:/Jonas/Research/Data/Campaigns/201411_Chile/DATA_ANALYSIS/Guallatiri/AnalysisResults/RT/Simulations/spectra/20160706_lightDilution/results/'
pTransCurves=r'D:/Jonas/Research/Data/Campaigns/201411_Chile/DATA_ANALYSIS/Guallatiri/AnalysisResults/RT/Suppl/Share/20160129_transcurves/'
transCurveFiles={"on"   :   "transcurve_onband_ecII_guallatiri.txt",
                 "off"   :   "transcurve_offband_ecII_guallatiri.txt"}

"""Function definitions
"""
def model(x,ia,i0, sigma):
    return i0*np.exp(-sigma*x)+ia*(1-np.exp(-sigma*x))

def fit(rads, dists, ia, guess=[None, 1e-4]):
    """Performs least square fit of data to model function à la Campion et al. 2015 
    """
    errFun=lambda p,x,y: (model(x,ia,*p)-y)**2
    if guess[0]==None:
        guess[0]=.05*ia
    return leastsq(errFun, guess, args=(dists,rads))


def fit_black(rads,dists,ia,guess=1e-4):
    """Performs least square fit of data to model function à la Campion et al. 2015 
    """
    errFun=lambda p,x,y: (model(x,ia,0,*p)-y)**2
    
    return leastsq(errFun, guess, args=(dists,rads))
    
def get_dist_and_tau(filename):
    spl=filename.split("_")
    tau=float(spl[1].split("Tau")[1])
    d=float(spl[2][3:8])
    return tau,d
    
def load_spectrum(filepath, dist):
    p = filepath
    dat=loadtxt(p)
    if p.endswith("out"):
        spec=Series(dat[:,4],dat[:,0])
        spec.dist=dist
    return spec
    
"""Here the evaluation script starts
"""

files=listdir(p)
specs=od()
for f in files:
    tau, d=get_dist_and_tau(f)
    if not specs.has_key(tau):
        specs[tau]=od()
    specs[tau][d]=load_spectrum(join(p,f), d)

bg=specs[0.0].values()[0]
#load and resample the transmission curves
tCurvesMerged=load_and_prepare_transcurves(transCurveFiles, pTransCurves, bg.index, removeBaseline=1)   
dists=np.asarray(specs[0.0].keys()[1:])*1000

vals=od()
ias=od()
results=od()
results_bad_guess=od()
results_black=od()

for run, spectra in specs.iteritems():
    tau=run
    print "\nReach new Tau: " + str(tau)
    bg=spectra.values()[0]
    specList=spectra.values()[1:]
    vals[tau]=od()
    ias[tau]=od()
    results[tau]=od()
    results_bad_guess[tau]=od()
    results_black[tau]=od()
    for k,v in tCurvesMerged.iteritems():
        filter=k
        print "Current filter: " + filter
        #ax2=v.plot(secondary_y=True, label="Filter trans " + k, mark_right=False )
        ia=np.nansum(bg*v)
        rads=[]
        for spec in specList:
            rad=np.nansum(spec*v)
            rads.append(rad)
            print "d=", spec.dist, ", Rad=", rad.mean()
            
        rads=np.asarray(rads)
        vals[tau][filter]=rads
        ias[tau][filter]=ia
        results[tau][filter]=fit(rads,dists,ia)
        results_bad_guess[tau][filter]=fit(rads,dists,ia, guess=[ia,1e-3])
        results_black[tau][filter]=fit_black(rads,dists,ia)
color.cycle_cmap(16, cmap="winter")
plt.close("all")
fig, axes= plt.subplots(2,2, figsize=(18,10), sharex=True)
    
def plot(dists, vals, ias,  aod=0.0, filter="on", resFree=None, resBlack=None, ax=None):
    if ax is None:
        fig, ax=plt.subplots(1,1)
    mag=dists.max()*.1
    dl=("Data, AOD: " + str(aod))
    p=ax.plot(dists, vals[aod][filter], "--o", label=dl)    
    x=np.linspace(-mag, dists.max()*1.1,100)
    
    lbl="Fit, AOD: " + str(aod) + " (I0 param free)" 
    if resFree is not None:    
        y=model(x,ias[aod][filter], resFree[aod][filter][0][0], resFree[aod][filter][0][1])
        ax.plot(x,y,"-",color=p[0].get_color(), label='_nolegend_')#label=lbl)
    if resBlack is not None:
        lbl="Fit, AOD: " + str(aod) + " (I0 param == 0)" 
        y=model(x,ias[aod][filter], 0, resBlack[aod][filter][0])
        ax.plot(x,y,"--",color=p[0].get_color(), label='_nolegend_')
    #ax.grid()
    return p,dl

# Change default color cycle for all new axes

for tau in vals.keys():    
    p,dl=plot(dists, vals, ias, tau,"on", resFree=results, resBlack=None, ax=axes[0,0])        
    plot(dists, vals, ias, tau,"off", resFree=results, resBlack=None, ax=axes[0,1])        
    plot(dists, vals, ias, tau,"on", resFree=None, resBlack=results_black, ax=axes[1,0])        
    plot(dists, vals, ias, tau,"off", resFree=None, resBlack=results_black, ax=axes[1,1])

    
axes[0,0].grid()    
axes[0,1].grid()
axes[1,0].grid()
axes[1,1].grid()
axes[0,0].set_ylabel("Measured radiance")
axes[1,0].set_ylabel("Measured radiance")
axes[1,0].set_xlabel("Distance [m]")
axes[1,1].set_xlabel("Distance [m]")
axes[0,0].set_title("On (I0 param free)")
axes[0,1].set_title("Off (I0 param free)")
axes[1,0].set_title("On (I0==0)")
axes[1,1].set_title("Off (I0==0)")
#plt.legend(ps, lbls)
fig.tight_layout()
fig.savefig(savep + "Fit results")
#axes[0,0].legend(loc="best", fancybox=True, framealpha=0.5, fontsize=10)
#axes[0,1].legend(loc='center right', bbox_to_anchor=(1.3, 0.5), fancybox=True, framealpha=0.5, fontsize=10).draggable()
axes[0,1].legend(loc='best',fancybox=True, framealpha=0.5, fontsize=10).draggable()

sigmas=od({"on"     :   [],
           "off"    :   []})
i0s=od({"on"     :   [],
        "off"    :   []})
           
sigmas_black=od({"on"     :   [],
                 "off"    :   []})
    
for result in results.values():
    if result["on"][1]==1:
        sigmas["on"].append(result["on"][0][1])
        i0s["on"].append(result["on"][0][0])
    else:
        sigmas["on"].append(np.nan)
        i0s["on"].append(np.nan)
    if result["off"][1]==1:
        sigmas["off"].append(result["off"][0][1])
        i0s["off"].append(result["off"][0][0])
    else:
        sigmas["off"].append(np.nan)
        i0s["off"].append(np.nan)
        
for result in results_black.values():
    sigmas_black["on"].append(result["on"][0])
    sigmas_black["off"].append(result["off"][0])
    
fig, axes=plt.subplots(1,3, figsize=(20,6))    
taus=results.keys()  
axes[0].plot(taus,sigmas["on"],"-xb", label="Ext coeffs. (I0 param free)")
axes[0].plot(taus,sigmas_black["on"],"--xg", label="Ext coeffs. (I0 param == 0)")
axes[0].legend(loc="best", fancybox=True, framealpha=0.5, fontsize=10)
axes[0].grid()
axes[0].set_title("On")
axes[0].set_xlabel("AOD")
axes[0].set_ylabel("Fitted extinction coeff")

axes[1].plot(taus,sigmas["off"],"-xb", label="Ext coeffs. (I0 param free)")
axes[1].plot(taus,sigmas_black["off"],"--xg", label="Ext coeffs. (I0 param == 0)")
axes[1].legend(loc="best", fancybox=True, framealpha=0.5, fontsize=10)
axes[1].grid()
axes[1].set_title("Off")
axes[1].set_xlabel("AOD")
axes[1].set_ylabel("Fitted extinction coeff")

r1=np.asarray(sigmas["on"])/np.asarray(sigmas["off"])
r2=np.asarray(sigmas_black["on"])/np.asarray(sigmas_black["off"])
axes[2].plot(taus,r1,"-xb", label="Ext coeffs. (I0 param free)")
axes[2].plot(taus,r2,"--xg", label="Ext coeffs. (I0 param == 0)")
axes[2].plot(axes[2].get_xlim(), (1.28413, 1.28413), 'r-', label="Pure Rayleigh")
axes[2].plot(axes[2].get_xlim(), (1.0645, 1.0645), 'c-', label="Mie law")
axes[2].legend(loc="best", fancybox=True, framealpha=0.5, fontsize=10)
axes[2].grid()
axes[2].set_title("Ratio On/Off")
axes[2].set_xlabel("AOD")
axes[2].set_ylabel("Ratio on/off")
fig.tight_layout()
fig.savefig(savep + "Extinction coeffs")