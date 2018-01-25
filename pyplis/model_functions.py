# -*- coding: utf-8 -*-
#
# Pyplis is a Python library for the analysis of UV SO2 camera data
# Copyright (C) 2017 Jonas Gliß (jonasgliss@gmail.com)
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License a
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
Pyplis module containing mathematical model functions
"""
from numpy import exp, sin, cos

# Polynomial fit functions of different order, including versions that go
# through the origin of the coordinate system
# (e.g. used in doascalib.py), dictionary keys are the polynomial order
polys = {1 : lambda x, a0, a1: a0*x + a1,
         2 : lambda x, a0, a1, a2: a0*x**2 + a1*x + a2,
         3 : lambda x, a0, a1, a2, a3: a0*x**3 + a1*x**2 + a2*x + a3}

polys_through_origin = {1 : lambda x, a0: a0*x,
                        2 : lambda x, a0, a1: a0*x**2 + a1*x,
                        3 : lambda x, a0, a1, a2: a0*x**3 + a1*x**2 + a2*x}

def get_poly_model(order=1, through_origin=False):
    """Helper method that returns a polynomial model function
    
    Parameters
    ----------
    order : int
        order of polynomial model (choose from 1-3)
    through_origin : bool
        if True, then the corresponding polynomial model function is returned
        that goes through the origin
    
    Returns
    -------
    lambda
        polynomial model function
    """
    if through_origin:
        if not polys_through_origin.has_key(order):
            raise ValueError("Polynomial model not available for input order "
                             "%.1f" %order)
        return polys_through_origin[order]
    if not polys.has_key(order):
        raise ValueError("Polynomial model not available for input order "
                             "%.1f" %order)
    return polys[order]

def dilutioncorr_model(dist, rad_ambient, i0, ext):
    """Model function for light dilution correction
    
    This model is based on the findings of `Campion et al., 2015 
    <http://www.sciencedirect.com/science/article/pii/S0377027315000189>`_.
    
    :param float dist: distance of dark (black) object in m
    :param float rad_ambient: intensity of ambient atmosphere at position of
        dark object
    :param float i0: initial intensity of dark object before it enters the 
        scattering medium. It is determined from the illumination intensity
        and the albedo of the dark object.
    :param float atm_ext: atmospheric scattering extincion coefficient 
        :math:`\epsilon` (in Campion et al., 2015 denoted with :math:`\sigma`).
        
    """
    return i0 * exp(-ext * dist) + rad_ambient * (1 - exp(-ext * dist))
    
def gaussian_no_offset(x, ampl, mu, sigma):
    """1D gauss with baseline zero
    
    :param float x: x position of evaluation
    :param float ampl: Amplitude of gaussian
    :param float mu: center poistion
    :param float sigma: standard deviation
    :returns float: value at position x    
    """
    #return float(ampl)*exp(-(x - float(mu))**2/(2*float(sigma)**2))
    return ampl * exp(-(x - mu)**2 / (2 * sigma**2))
    
def gaussian(x, ampl, mu, sigma, offset):
    """1D gauss with arbitrary baseline
    
    :param float x: x position of evaluation
    :param float ampl: Amplitude of gaussian
    :param float mu: center poistion
    :param float sigma: standard deviation
    :param float offset: baseline of gaussian
    :returns float: value at position x
    """
    return gaussian_no_offset(x, ampl, mu, sigma) + offset
    
def multi_gaussian_no_offset(x, *params):
    """Superimposed 1D gauss functions with baseline zero
    
    :param array x: x array used for evaluation
    :param list *params: List of length L = 3xN were N corresponds to the 
        number of gaussians e.g.::
        
            [100,10,3,50,15,6]
            
        would correspond to 2 gaussians with the following characteristics:
        
            1. Peak amplitude: 100, Mu: 10, sigma: 3
            2. Peak amplitude: 50, Mu: 15, sigma: 6
    """
    res = 0
    num = len(params) / 3
    for k in range(num):
        p = params[k*3:(k+1)*3]
        res = res + gaussian_no_offset(x,*p)
    return res

def multi_gaussian_same_offset(x, offset, *params):
    """Superimposed 1D gauss functions with baseline (offset)
    
    See :func:`multi_gaussian_no_offset` for instructions
    """
    return multi_gaussian_no_offset(x, *params) + offset
        
def supergauss_2d((x, y), amplitude, xm, ym, sigma, asym, shape, offset):
    """2D super gaussian without tilt
    
    :param tuple (x, y): position 
    :param float amplitude: amplitude of peak
    :param float xm: x position of maximum
    :param float ym: y position of maximum
    :param float asym: assymetry in y direction (1 is circle, smaller 
            means dillated in y direction)
    :param float shape: super gaussian shape parameter (1 is gaussian)
    :param float offset: base level of gaussian 
    """
    u = ((x - xm) / sigma) ** 2 + ((y - ym) * asym / sigma)**2
    g = offset + amplitude * exp(-u**shape)
    return g.ravel()
    
def supergauss_2d_tilt((x, y), amplitude, xm, ym, sigma, asym, shape, offset,\
                                                                        theta):
    """2D super gaussian without tilt
    
    :param tuple (x, y): position
    :param float amplitude: amplitude of peak
    :param float xm: x position of maximum
    :param float ym: y position of maximum
    :param float asym: assymetry in y direction (1 is circle, smaller 
            means dillated in y direction)
    :param float shape: super gaussian shape parameter (2 is gaussian)
    :param float offset: base level of gaussian 
    :param float theta: tilt angle (rad) of super gaussian
    
    """
    xprime = (x - xm) * cos(theta) - (y - ym) * sin(theta)
    yprime = (x - xm) * sin(theta) + (y - ym) * cos(theta)
    u = (xprime / sigma)**2 + (yprime * asym / sigma)**2
    g = offset + amplitude * exp(-u**shape)
    return g.ravel()
