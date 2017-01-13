Removed from article
####################

.. note::

    Collections of paragraphs which were removed from article which might be useful for the docs at the appropriate positions

Geometrical calculations
------------------------   
If all relevant information is available a :class:`geonum.base.GeoSetup` instance is created within :class:`piscope.Utils.MeasGeometry` which (by default) consists of three :class:`geonum.base.GeoPoint` objects (cam, source, intersect) and 3 :class:`geonum.base.GeoVector3D` objects (cfov, camVec, plumeVec) where the the intersect point refers to the coordinates where the camera CFOV intersects the plume vector. These objects can be accessed via camera and plume coordinates and relevant directions.

Note that in :mod:`geonum` azimuth angles are defined $-180^{\circ} \leq \theta \leq 180^{\circ}$ and in piSCOPE $0^{\circ} \leq \theta \leq 360^{\circ}$ where $0^{\circ}$ corresponds to north direction.
  
