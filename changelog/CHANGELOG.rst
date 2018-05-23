Not yet released (> v1.3.0)
=======================================

1.3.0 -> 1.3.1
--------------

- Removed strict dependency for pillow (v1.3.1)

  In order to reduce the number of dependencies, the default image read function is ``cv2.imread`` from the OpenCV library. Before, the ``imread`` function of matplotlib was used which uses the
  ``pillow`` (or formerly ``PIL``) library.

  Also, it appeared complicated to install pillow using conda on a Windows machine (see installation instructions below).
