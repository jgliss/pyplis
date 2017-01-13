Preface
-------

piSCOPE is a Python library for the analysis of image data from UV SO2 cameras (and other scientific cameras working on the same measurement principle). Such cameras are ususally pointed into the emission plume of a located source emitter (e.g. volcanoes, power plants, ships) in order to measure SO2 emission rates of the source. The measurement principle is based on absorption spectroscopy of scattered sunlight in the ultraviolet wavelength range below 340 nm. Such cameras usually measure in two wavelength bands (of about 10 nm width) using optical bandpass filters. One filter (SO2 on band) is situated around 310 nm where SO2 shows distsinct absorption features and one around 330 nm where SO2 absorption is weak. Optical densities (ODs) of the plume :math:`\tau = \ln\left(\frac{I_0}{I}\right)` are retrieved relative to the sky background intensity :math:`I_0` in each image pixel. The on band ODs are dominated by SO2 absorption but may also include further optical densities, for instance due to aerosole scattering in the plume. The latter phenomenon is of broadband nature (as other scattering phenomena) and can be corrected for by using the off band ODs and determining the apparent absorbance of SO2:

.. math::

  \tau_\text{AA} = \tau_{\text{on}}\,-\,\tau_{\text{off}} = \ln\left(\frac{I_0}{I}\right)_{\text{on}}-\ln\left(\frac{I_0}{I}\right)_{\text{off}}
  
:math:`\tau_\text{AA}` images are converted into SO2 column density (CD) images where 

.. math::

  S_{SO2}(i,j)=\int_{\mathcal{S}} c(x,y,z) ds 

denotes the SO2 column density along the viewing direction :math:`\mathcal{S}` of image pixel :math:`(i,j)`, and :math:`c(x,y,z)` is the concentration distribution of SO2.

Emission rates are retrieved by integrating the SO2 CDs along a plume cross section line :math:`\bm{\ell}` which should stand approximately perpendicular to the plume propagation direction in the image and should cover a whole plume cross section. The emission rate :math:`\Phi` through :math:`\bm{\ell}` is then determined by multiplication of the integrated columns with the orjected plume speeds :math:`\bm{\bar{v}}_{ij}` along 
:math:`\bm{\ell}`:
    
.. math::

  \Phi=f^{-1}\sum_{m=1}^{M}S_\text{SO2}(m)\cdot\left\langle\bm{\bar{v}}_{ij}(m)\cdot\bm{\hat{n}}(m)\right\rangle\cdot d_\text{pl}(m)\cdot\Delta s(m)
  
where *m* denotes one of a total of *M* sample positions along :math:`\bm{\ell}` in the image plane. :math:`\Delta s` is the integration step (measured in physical distances on the detector). *f* is the focal length of the camera, :math:`d_\text{pl}` the plume distance and :math:`\bm{\hat{n}}` is the normal of :math:`\bm{\ell}`. The integration step :math:`\Delta s` as well as the plume distances :math:`d_\text{pl}` can be derived from the measurement geometry and require a minimum information about camera and source position, viewing direction, optics and meteorological wind direction. Plume speeds are usually retrieved from the images directly either using cross correlation based methods or motion estimation algorithms, for instance dense optical flow algorithms.

piSCOPE is designed in a modular architecture and includes routines for all required analysis steps related to emission rate retrievals. The basic datastructure is organised in a hierarchical structure:

.. thumbnail::  ../data/illustrations/flowchart_setup.png
   
   Flowchart of basic data structure including setup classes (light orange), Dataset classes (light blue) and image list classes (light green) and required meta information (white)

They include:

  1. Setup classes including / defining measurement meta information (:mod:`piscope.setupclasses`)
  #. The Dataset object (:class:`piscope.dataset.Dataset`, note that the :class:`piscope.calibration.CellCalib` inherits from Dataset and can be used as such)
  #. Image list objects (:mod:`piscope.imglists`)
  #. The Img object (:class:`piscope.image.Img`)


The most important analysis routines are organised in the following modules:

  1. Detailed analysis of measurement geometry (:mod:`piscope.geometry`)
  #. Engines to perform camera calibration (:mod:`piscope.calibration`) including functionality for DOAS FOV search(:mod:`piscope.doasfov`)
  #. Plume speed analysis using optical flow or cross correlation method (:mod:`piscope.plumespeed`)
  
    
  
.. todo::

  1. Insert flowcharts for basic data structure
  2. Insert 2D / 3D sketch of measurement setup 

