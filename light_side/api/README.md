# APIs <!-- omit in toc -->

- [Available Models](#available-models)
- [Available Versions for a Spesific Model](#available-versions-for-a-spesific-model)
- [Latest Version for a Spesific Model](#latest-version-for-a-spesific-model)
- [Pretrained Model](#pretrained-model)
- [Model with Random Weight Initialization](#model-with-random-weight-initialization)
- [Pretrained Arch Model](#pretrained-arch-model)
- [Arch Model with Random Weight Initialization](#arch-model-with-random-weight-initialization)

## Available Models

```python
import light_side as ls
ls.available_models()
# ['zerodce_3-32-16_zerodce', 'zerodce_7-16-8_zerodce', 'zerodce_7-32-16_zerodce', 'zerodce_7-32-8_zerodce']
```

## Available Versions for a Spesific Model

```python
import light_side as ls
model_name = 'zerodce_3-32-16_zerodce'
ls.get_model_versions(model_name)
# ['0', '1']
```

## Latest Version for a Spesific Model

```python
import light_side as ls
model_name = 'zerodce_3-32-16_zerodce'
ls.get_model_latest_version(model_name)
# '0'
```

## Pretrained Model

```python
import light_side as ls
model_name = 'zerodce_3-32-16_zerodce'
model = ls.Enhancer.from_pretrained(model_name, version=None) # if version none is given than latest version will be used.
# model: pl.LightningModule
```

## Model with Random Weight Initialization

```python
import light_side as ls
arch = 'zerodce'
config = '3-32-16'
model = ls.Enhancer.build(arch, config)
# model: pl.LightningModule
```

## Pretrained Arch Model

```python
import light_side as ls
model_name = 'zerodce_3-32-16_zerodce'
model = ls.Enhancer.from_pretrained_arch(model_name, version=None) # if version none is given than latest version will be used.
# model: torch.nn.Module
```

## Arch Model with Random Weight Initialization

```python
import light_side as ls
arch = 'zerodce'
config = '3-32-16'
model = ls.Enhancer.build_arch(arch, config)
# model: torch.nn.Module
```
