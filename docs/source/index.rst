.. light_side documentation master file, created by
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
:|:clapper:| Pages:
   - |:film_projector:| `Project Page <https://canturan10.github.io/light_side>`_
   - |:camera:| `Github Page <https://github.com/canturan10/light_side>`_
   - |:camera_flash:| `Hugging Face Demo Page <https://huggingface.co/spaces/canturan10/light_side>`_

.. toctree::
   :maxdepth: 2
   :name: starter
   :caption: Getting Started

   starter/about.md
   starter/prerequisites.md
   starter/installation.md
   starter/apis.md
   starter/archs.md
   starter/datasets.md
   starter/deployment.md

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

   deployment/fastapi.rst
   deployment/onnx_export.rst
   deployment/onnx_runtime.rst
   deployment/deepsparse.rst
   deployment/tensorflow_export.rst
   deployment/tensorflow_runtime.rst
   deployment/tensorflow_lite_export.rst
   deployment/tensorflow_lite_runtime.rst