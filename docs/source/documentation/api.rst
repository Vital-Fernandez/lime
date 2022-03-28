.. _api:

API
===

Inputs and outputs
------------------

.. autofunction:: lime.load_cfg

.. autofunction:: lime.load_lines_log

.. autofunction:: lime.save_param_maps


Core
----

.. autoclass:: lime.Spectrum
   :members: fit_from_wavelengths, match_line_mask, display_results, plot_spectrum

.. autofunction:: lime.MaskInspector


Convenience functions
---------------------

.. autofunction:: lime.label_decomposition

.. autofunction:: lime.spatial_mask_generator

.. autofunction:: lime.CubeInspector
