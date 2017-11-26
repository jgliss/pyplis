*******
Preface
*******

.. include:: ../README.rst
  
Flowchart main analysis steps
=============================

The following flowchart illustrates the main analysis steps for emission-rate retrievals using UV SO2 cameras. The colours indicate geometrical calculations (yellow), background modelling (light gray), camera calibration (light blue), plume speed retrieval (light purple) and the central processing steps for the emission-rate retrieval (light green). Shaded and dashed symbols indicate optional or alternative analysis methods.

.. thumbnail::  ./_graphics/01_flowchart_physical.png
  :title:
  
  Flowchart showing the main analysis steps for emission rate retrievals
  
  
Flowchart code hierarchy
========================

The following flowchart illustrates the most relevant classes / methods of the *Pyplis* API with a focus on the required routines for SO2 emission-rate retrievals. Italic denotations correspond to class names in Pyplis. Optional / alternative analysis procedures are indicated by dashed boxes. Setup classes (red) include relevant meta information and can be used to create Dataset objects (blue). The latter perform file separation by image type and create ImgList objects (green) for each type (e.g. on, off, dark). Further analysis classes are indicated in yellow. Note that the routine for signal dilution correction is not shown here.

.. thumbnail::  ./_graphics/02_flowchart_datastructure.png
  :title:
  
  Flowchart illustrating the basic architecture of pyplis (note: the engine for signal dilution correction is not included here).
  
