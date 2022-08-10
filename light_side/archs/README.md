[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zero-reference-deep-curve-estimation-for-low/low-light-image-enhancement-on-dicm)](https://paperswithcode.com/sota/low-light-image-enhancement-on-dicm?p=zero-reference-deep-curve-estimation-for-low)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zero-reference-deep-curve-estimation-for-low/low-light-image-enhancement-on-lime)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lime?p=zero-reference-deep-curve-estimation-for-low)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zero-reference-deep-curve-estimation-for-low/low-light-image-enhancement-on-mef)](https://paperswithcode.com/sota/low-light-image-enhancement-on-mef?p=zero-reference-deep-curve-estimation-for-low)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zero-reference-deep-curve-estimation-for-low/low-light-image-enhancement-on-npe)](https://paperswithcode.com/sota/low-light-image-enhancement-on-npe?p=zero-reference-deep-curve-estimation-for-low)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zero-reference-deep-curve-estimation-for-low/low-light-image-enhancement-on-vv)](https://paperswithcode.com/sota/low-light-image-enhancement-on-vv?p=zero-reference-deep-curve-estimation-for-low)

# Architectures  <!-- omit in toc -->

- [ZeroDCE](#zerodce)
- [Citation](#citation)

## ZeroDCE

**Zero-DCE** (Zero-Reference Deep Curve Estimation), which formulates light enhancement as a task of image-specific curve estimation with a deep network. Our method trains a lightweight deep network, DCE-Net, to estimate pixel-wise and high-order curves for dynamic range adjustment of a given image. The curve estimation is specially designed, considering pixel value range, monotonicity, and differentiability. Zero-DCE is appealing in its relaxed assumption on reference images, i.e., it does not require any paired or unpaired data during training. This is achieved through a set of carefully formulated non-reference loss functions, which implicitly measure the enhancement quality and drive the learning of the network. Our method is efficient as image enhancement can be achieved by an intuitive and simple nonlinear curve mapping. Despite its simplicity, we show that it generalizes well to diverse lighting conditions. Extensive experiments on various benchmarks demonstrate the advantages of our method over state-of-the-art methods qualitatively and quantitatively. Furthermore, the potential benefits of our Zero-DCE to face detection in the dark are discussed. Code and model will be available at <https://github.com/Li-Chongyi/Zero-DCE>.

|  Architecture   | Configuration | Parameters | Model Size |
| :-------------: | :-----------: | :--------: | :--------: |
| **ZeroDCE** |    3-32-16    |   24.0 K    |  0.096 MB  |
| **ZeroDCE** |    7-16-8    |   23.6 K    |  0.095 MB  |
| **ZeroDCE** |    7-32-8    |   79.5 K    |  0.318 MB  |
| **ZeroDCE** |    7-32-16    |   79.5 K    |  0.318 MB  |

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2001-06826,
  author    = {Chunle Guo and
               Chongyi Li and
               Jichang Guo and
               Chen Change Loy and
               Junhui Hou and
               Sam Kwong and
               Runmin Cong},
  title     = {Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement},
  journal   = {CoRR},
  volume    = {abs/2001.06826},
  year      = {2020},
  url       = {https://arxiv.org/abs/2001.06826},
  eprinttype = {arXiv},
  eprint    = {2001.06826},
  timestamp = {Sat, 23 Jan 2021 01:20:17 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2001-06826.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
