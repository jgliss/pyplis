***************
Example scripts
***************

pyplis example scripts. The scripts require downloading the `pyplis test data <http://pyplis.readthedocs.io/en/latest/intro.html#example-and-test-data>`__.

.. note::

  The scripts are based on the latest commit in the `GitHub repo <https://github.com/jgliss/pyplis>`__. If you have installed an older version of pyplis, please use the corresponding scripts which are provided `here <https://github.com/jgliss/pyplis/tree/master/scripts/old_versions>`__.

Introductory scripts
===================

These scripts give an introduction into basic features and classes of pyplis.

.. _ex01:

Example 0.1 - Image representation
----------------------------------

Introduction into :class:`Img` object and basic usage including correction for dark current and detector offset.

**Code**

.. literalinclude:: ../scripts/ex0_1_img_handling.py

.. _ex02:

Example 0.2 - The camera class
------------------------------

Introduction into the :class:`Camera` object using the example of the ECII camera standard.

**Code**

.. literalinclude:: ../scripts/ex0_2_camera_setup.py

.. _ex03:

Example 0.3 - Introduction into ImgList objects
-----------------------------------------------

Manual creation of :class:`ImgList` objects and basic features.

**Code**

.. literalinclude:: ../scripts/ex0_3_imglists_manually.py

.. _ex04:

Example 0.4 - Introduction into the Dataset class
-------------------------------------------------

Automatic image type separation using the :class:`Dataset` object and the ECII camera standard.

**Code**

.. literalinclude:: ../scripts/ex0_4_imglists_auto.py

.. _ex05:

Example 0.5 - Optical flow live view
------------------------------------

Live view of optical flow calculation using :class:`OpticalFlowFarneback` object (requires webcam).

**Code**

.. literalinclude:: ../scripts/ex0_5_optflow_livecam.py

.. _ex06:

Example 0.6 - Plume cross section lines
---------------------------------------

Creation and orientation of :class:`LineOnImage` objects for emission rate retrievals.

**Code**

.. literalinclude:: ../scripts/ex0_6_pcs_lines.py

.. _ex07:

Example 0.7 - Manual cell calibration
-------------------------------------

Manual cell calibration based on a set of background and cell images (on / off).

**Code**

.. literalinclude:: ../scripts/ex0_7_cellcalib_manual.py


Examples for emission rate analysis
===================================

The following scripts are directly related for emission rate analysis and build on top of each other ending with a full emission rate analysis in :ref:`ex12`.

.. _ex1:

Example 1 - Creation of analysis setup and Dataset
--------------------------------------------------

This script introduces the :class:`MeasSetup` object and how it can be used to specify all relevant information to create a :class:`Dataset` for emission rate analysis.

**Code**

.. literalinclude:: ../scripts/ex01_analysis_setup.py

.. _ex2:

Example 2 - Measurement Geometry
--------------------------------

This script introduces the :class:`MeasGeometry` object including some basic features.

**Code**

.. literalinclude:: ../scripts/ex02_meas_geometry.py

.. _ex3:

Example 3 - Plume background analysis
-------------------------------------

Introduction into :class:`PlumeBackgroundModel` object the default modes for retrieval of plume background intensities and plume optical density images.

**Code**

.. literalinclude:: ../scripts/ex03_plume_background.py

.. _ex4:

Example 4 - Preparation of AA image list
----------------------------------------

Script showing how to prepare an :class:`ImgList` containing on-band plume images, such that ``aa_mode`` can be activated (i.e. images are loaded as AA images).

**Code**

.. literalinclude:: ../scripts/ex04_prep_aa_imglist.py

.. _ex5:

Example 5 - Automatic cell calibration
--------------------------------------

This scripts shows how to perform automatic cell calibration based on a time series of on and off band images containing both, suitable background images and images from different SO2 cells (in both wavelength channels, cf. :ref:`ex07`).

**Code**

.. literalinclude:: ../scripts/ex05_cell_calib_auto.py

.. _ex6:

Example 6 - DOAS calibration
----------------------------

Introduction into DOAS calibration including FOV search using both, the Pearson and the IFR method.

**Code**

.. literalinclude:: ../scripts/ex06_doas_calib.py

.. _ex7:

Example 7 - AA sensitivity correction masks
-------------------------------------------

Combine the results from :ref:`ex5` and :ref:`ex6` in order to retrieve AA sensitivity correction masks normalised to the position of the DOAS FOV.

**Code**

.. literalinclude:: ../scripts/ex07_doas_cell_calib.py

.. _ex8:

Example 8 - Plume velocity retrieval (Cross correlation)
--------------------------------------------------------

In this script an exemplary plume velocity retrieval is performed using the signal cross correlation algorithm. The velocity is retrieved based on two plume cross sections and a time series of plume AA images (using the AA :class:`ImgList` created in :ref:`ex4`).

**Code**

.. literalinclude:: ../scripts/ex08_velo_crosscorr.py

.. _ex9:

Example 9 - Plume velocity retrieval (Optical flow Farneback)
-------------------------------------------------------------

This script gives an introduction into plume velocity retrievals using the Farneback optical flow algorithm (:class:`OpticalFlowFarneback`) and a histogram based post analysis.

**Code**

.. literalinclude:: ../scripts/ex09_velo_optflow.py

.. _ex10:

Example 10 - Import plume background images
-------------------------------------------

Create a :class:`Dataset` object for a time interval containing only plume background images (on / off).

.. note::

  Stand alone script that is not required for any of the following scripts

**Code**

.. literalinclude:: ../scripts/ex10_bg_imglists.py

.. _ex11:

Example 11 - Image based signal dilution correction
---------------------------------------------------

This script introduces the image based signal dilution correction including automatic retrieval of terrain distances on a pixel basis.

**Code**

.. literalinclude:: ../scripts/ex11_signal_dilution.py

.. _ex12:

Example 12 - Emission rate analysis (Etna example data)
-------------------------------------------------------

Perform emission rate analysis for the example data. The analysis is performed along one plume cross section (in the image center) and using three different plume velocity retrievals.

**Code**

.. literalinclude:: ../scripts/ex12_emission_rate.py
