LiMe: A Line Measuring library
================================

.. image:: ./_static/logo_transparent.png
    :scale: 30%
    :align: center

This library provides a set of tools to fit lines in astronomical spectra. Its design aims for a user-friendly workflow
for both single lines and large data sets. The library provides tools for masking, detecting, profile fitting
and storing the results. The output measurements support both the posterior analysis of the object's chemical composition
and plasma kinematics.

These are some of the features currently available:

* Non-profile and Gaussian profile emission and absorption line measurements.
* The user can include the pixel error spectrum in the calculation.
* The Multi-Gaussian profile parameters can be constrained by the user during the fitting.
* Tools to confirm the presence of lines.
* Static and interactive plots for the visual appraisal of inputs and outputs
* Line labels adhere to the `PyNeb <http://research.iac.es/proyecto/PyNeb/>`_ format.
* The measurements can be saved in several file types, including multi-page *.fits*, *.asdf* and *.xlsx* files

.. admonition:: Where to find what you need
   :class: hint

   💻 To install or update the library go to the `installation page <documentation/installation.html>`_. Download the
   `sample data folder <https://github.com/Vital-Fernandez/lime/tree/master/examples/sample_data>`_ and try to run the
   `examples <https://github.com/Vital-Fernandez/lime/tree/master/examples>`_.

   🚀 For a quick start go to the **Tutorials** section. These are organized by increasing complexity and they
   provide a working knowledge of the library algorithms.

   🌀 To learn more about the library design check the :ref:`inputs <inputs>` and :ref:`profile fitting <profileFitting>`
   documentation. The library functions description can be found at the :ref:`API <api>`.

   📈 The outputs physical description is available in the :ref:`measurements <measurements_page>` documentation.

.. :ref:`doc-tree`

.. toctree::
   :maxdepth: 1
   :caption: Documentation
   :name: doc-tree

   documentation/installation
   documentation/fitting
   documentation/plots
   documentation/measurements
   documentation/api

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :name: tutorial-tree

   tutorials/1_SingleLineFitting
   tutorials/2_Line_bands_inspection
   tutorials/3_CompleteSpectrumFitting
   tutorials/4_IFU_spatial_masking
   tutorials/5_IFU_line_fitting
   tutorials/6_IFU_review_results
   tutorials/7_SyntheticObservation

