# Deployments <!-- omit in toc -->

- [ONNX](#onnx)
	- [Install Dependency For ONNX](#install-dependency-for-onnx)
	- [Convert Model to ONNX](#convert-model-to-onnx)
	- [ONNX Runtime](#onnx-runtime)

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
