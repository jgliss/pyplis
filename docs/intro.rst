*************
Introduction
*************

.. include:: ../README.rst

Visualisation of API architecture
=================================

The following two flowcharts illustrate details about the Pyplis architecture, for details, please see section :ref:`article`.

Flowchart emission-rate retrieval (scientific)
----------------------------------------------

The following flowchart illustrates the main analysis steps for emission-rate retrievals using UV SO2 cameras. The colours indicate geometrical calculations (yellow), background modelling (light grey), camera calibration (light blue), plume speed retrieval (light purple) and the central processing steps for the emission-rate retrieval (light green). Shaded and dashed symbols indicate optional or alternative analysis methods.

.. figure::  ./_graphics/flowchart_physical.png
  :width: 80%
  :align: center

  Flowchart showing the main analysis steps for emission rate retrievals


Flowchart API (code architecture)
---------------------------------

The following flowchart illustrates the most relevant classes / methods of the *Pyplis* API with a focus on the required routines for SO2 emission-rate retrievals. Italic denotations correspond to class names in Pyplis. Optional / alternative analysis procedures are indicated by dashed boxes. Setup classes (red) include relevant meta information and can be used to create Dataset objects (blue). The latter perform file separation by image type and create ImgList objects (green) for each type (e.g. on, off, dark). Further analysis classes are indicated in yellow. Note that the routine for signal dilution correction is not shown here.

.. figure::  ./_graphics/flowchart_datastructure.png
  :width: 80%
  :align: center

  Flowchart illustrating the basic architecture of pyplis (note: the engine for signal dilution correction is not included here).
