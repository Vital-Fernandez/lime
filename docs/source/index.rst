LiMe: A Line Measuring library
================================

.. image:: ./_static/logo_transparent.png
    :scale: 30%
    :align: center

This library provides a set of tools to fit lines in astronomical spectra. Its design aims for a user-friendly workflow
for both single lines and Big Data observations. The library provides tools for masking, detecting and fitting lines, as
well as, storing the results. The measurements support the posterior analysis of the object chemical composition and plasma
kinematics.

These are some of the features currently available:

* Non-profile-dependence and Gaussian profile measurements.
* The user can include the pixel error spectrum in the calculation.
* Multi-Gaussian profile fitting with flexible definition for the parameters boundaries.
* Tools to confirm the presence of lines.
* Static and interactive plots for the visual appraisal of inputs and outputs
* Emission line labels adhere to the `PyNeb <http://research.iac.es/proyecto/PyNeb/>`_ format.
* The measurements can be saved in several formats, including multi-page *.fits*, *.asdf* and *.xlsx* files

.. admonition:: Where to find what you need
   :class: hint

   💻 To install or update the library go to the `installation page <documentation/installation.html>`_. Download the
   `sample data folder <https://github.com/Vital-Fernandez/lime/tree/master/examples/sample_data>`_ and try to run the
   `examples <https://github.com/Vital-Fernandez/lime/tree/master/examples>`_.

   🚀 For a quick start go to the **Tutorials** section. These are organized by increasing complexity and they
   provide a working knowledge of the library algorithms.

   🌀 To learn more about the library design check the :ref:`inputs <inputs>` and :ref:`profile fitting <profileFitting>`
   documentation. The library functions manual is found at the :ref:`API <api>`.

   📈 The outputs physical description can be found in the :ref:`measurements <measurements_page>` documentation.

.. :ref:`doc-tree`

.. toctree::
   :maxdepth: 1
   :caption: Documentation
   :name: doc-tree

   documentation/installation
   documentation/fitting
   documentation/measurements
   documentation/api

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :name: tutorial-tree

   tutorials/1_BasicFit
   tutorials/2_Synthetic
   tutorials/3_CompleteSpec
   tutorials/4_Mask_selection
   tutorials/5_IFU_masks
   tutorials/6_IFU_fittings
   tutorials/7_IFU_results
   tutorials/8_LiMe_Notebooks

