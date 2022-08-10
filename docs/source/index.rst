.. satellighte documentation master file, created by
   sphinx-quickstart on Sat Feb 19 00:32:16 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|:zap:| Light Side Documentation
=============================================

Low-Light Image Enhancement
---------------------------------------------

**Light Side** is an low-light image enhancement library  that consist state-of-the-art deep learning methods. The light side of the Force is referenced. The aim is to create a light structure that will find the `Light Side of the Night`.

:|:zap:| Pypi: `light_side <https://pypi.org/project/light_side/>`_
:|:flying_saucer:| Version: |release|

.. toctree::
   :maxdepth: 2
   :name: starter
   :caption: Getting Started

   starter/about.md
   starter/prerequisites.md
   starter/installation.md
   starter/archs.md
   starter/datasets.md

.. toctree::
   :maxdepth: 1
   :name: api
   :caption: Light Side API

   api/api.rst
   api/module.rst
   api/datasets.rst

.. toctree::
   :maxdepth: 1
   :name: deployment
   :caption: Deployment

   deployment/onnx_export.rst
   deployment/onnx_runtime.rst