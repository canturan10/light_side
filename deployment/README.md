# Deployments <!-- omit in toc -->

- [ONNX](#onnx)
	- [Install Dependency For ONNX](#install-dependency-for-onnx)
	- [Convert Model to ONNX](#convert-model-to-onnx)
	- [ONNX Runtime](#onnx-runtime)
- [DeepSparse](#deepsparse)
	- [Install Dependency For DeepSparse](#install-dependency-for-deepsparse)
	- [DeepSparse Runtime](#deepsparse-runtime)

## ONNX

[ONNX Runtime](https://onnxruntime.ai/) inference can lead to faster customer experiences and lower costs.

From root directory of the repository run followings,

### Install Dependency For ONNX

```bash
pip install onnx~=1.11.0
pip install onnxruntime~=1.10.0
```

### Convert Model to ONNX

```bash
python deployment/onnx/export.py
```

### ONNX Runtime

```bash
python deployment/onnx/runtime.py
```

## DeepSparse

Neural Magic's [DeepSparse](https://docs.neuralmagic.com/deepsparse/) Engine is able to integrate into popular deep learning libraries allowing you to leverage DeepSparse for loading and deploying sparse models with ONNX.

From root directory of the repository run followings. We need the `ONNX` model to use it. [Create your onnx model from the above steps](#onnx). Next,

### Install Dependency For DeepSparse

```bash
pip install deepsparse~=1.0.2
```

### DeepSparse Runtime

```bash
python deployment/deepsparse/runtime.py
```
