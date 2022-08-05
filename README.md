<p align="right">
    <a href="https://www.buymeacoffee.com/canturan10"><img src="https://img.buymeacoffee.com/button-api/?text=You can buy me a coffee&emoji=&slug=canturan10&button_colour=5F7FFF&font_colour=ffffff&font_family=Comic&outline_colour=000000&coffee_colour=FFDD00" width="200" /></a>
</p>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zero-reference-deep-curve-estimation-for-low/low-light-image-enhancement-on-dicm)](https://paperswithcode.com/sota/low-light-image-enhancement-on-dicm?p=zero-reference-deep-curve-estimation-for-low)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zero-reference-deep-curve-estimation-for-low/low-light-image-enhancement-on-lime)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lime?p=zero-reference-deep-curve-estimation-for-low)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zero-reference-deep-curve-estimation-for-low/low-light-image-enhancement-on-mef)](https://paperswithcode.com/sota/low-light-image-enhancement-on-mef?p=zero-reference-deep-curve-estimation-for-low)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zero-reference-deep-curve-estimation-for-low/low-light-image-enhancement-on-npe)](https://paperswithcode.com/sota/low-light-image-enhancement-on-npe?p=zero-reference-deep-curve-estimation-for-low)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zero-reference-deep-curve-estimation-for-low/low-light-image-enhancement-on-vv)](https://paperswithcode.com/sota/low-light-image-enhancement-on-vv?p=zero-reference-deep-curve-estimation-for-low)

<!-- PROJECT SUMMARY -->
<p align="center">
    <img width="100px" src="https://raw.githubusercontent.com/canturan10/light_side/master/src/light_side.png" align="center" alt="Light Side" />
<h2 align="center">Light Side of the Night</h2>
<h4 align="center">Low-Light Image Enhancement</h4>

<!--
<p align="center">
    <strong>
        <a href="https://canturan10.github.io/light_side/">Website</a>
        •
        <a href="https://light_side.readthedocs.io/">Docs</a>
        •
        <a href="https://share.streamlit.io/canturan10/light_side-streamlit/app.py">Demo</a>
    </strong>
</p>
-->

<!-- TABLE OF CONTENTS -->
<details>
    <summary>
        <strong>
            TABLE OF CONTENTS
        </strong>
    </summary>
    <ol>
        <li>
            <a href="#about-the-light-side">About The Light Side</a>
        </li>
        <li>
            <a href="##prerequisites">Prerequisites</a>
        </li>
        <li>
            <a href="#installation">Installation</a>
            <ul>
                <li><a href="#from-pypi">From Pypi</a></li>
                <li><a href="#from-source">From Source</a></li>
            </ul>
        </li>
        <li><a href="#usage-examples">Usage Examples</a></li>
        <li><a href="#architectures">Architectures</a></li>
        <li><a href="#datasets">Datasets</a></li>
        <li><a href="#deployments">Deployments</a></li>
        <li><a href="#training">Training</a></li>
        <li><a href="#tests">Tests</a></li>
        <li><a href="#contributing">Contributing</a></li>
        <li><a href="#contributors">Contributors</a></li>
        <li><a href="#contact">Contact</a></li>
        <li><a href="#license">License</a></li>
        <li><a href="#references">References</a></li>
        <li><a href="#citations">Citations</a></li>
    </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Light Side

**Light Side** is an low-light image enhancement library  that consist state-of-the-art deep learning methods. The light side of the Force is referenced. The aim is to create a light structure that will find the `Light Side of the Night`.

> <img width="80px" src="https://raw.githubusercontent.com/canturan10/light_side/master/src/Light_side_of_the_Force.png" align="left" style="padding: 0px 20px ;" alt="Light_side_of_the_Force"/> **The light side of the Force**, also known as Ashla, was one of two methods of using the Force. The light side was aligned with calmness, peace, and passiveness, and was used only for knowledge and defense. The Jedi were notable practitioners of the light, being selfless servants of the will of the Force, and their enemies, the Sith followed the dark side of the Force.
>
> _Source: [Wookieepedia](https://starwars.fandom.com/wiki/Light_side_of_the_Force)_

> **Low-light image enhancement** aims at improving the perception or interpretability of an image captured in an environment with poor illumination.
>
> _Source: [paperswithcode](https://paperswithcode.com/task/low-light-image-enhancement)_

<!-- PREREQUISITES -->
## Prerequisites

Before you begin, ensure you have met the following requirements:

| requirement       | version  |
| ----------------- | -------- |
| imageio           | ~=2.15.0 |
| numpy             | ~=1.21.0 |
| pytorch_lightning | ~=1.6.0  |
| scikit-learn      | ~=1.0.2  |
| torch             | ~=1.8.1  |

<!-- INSTALLATION -->
## Installation

To install Light Side, follow these steps:

### From Pypi

```bash
pip install light_side
```

### From Source

```bash
git clone https://github.com/canturan10/light_side.git
cd light_side
pip install .
```

#### From Source For Development

```bash
git clone https://github.com/canturan10/light_side.git
cd light_side
pip install -e ".[all]"
```
<!-- USAGE EXAMPLES -->
## Usage Examples

```python
import imageio
import light_side as ls

img = imageio.imread("test.jpg")

model = ls.Enhancer.from_pretrained("model_config_dataset")
model.eval()

results = model.predict(img)
```

<!-- ARCHITECTURES -->
## Architectures

- [x] [Zero DCE](https://github.com/canturan10/light_side/blob/master/light_side/archs/README.md)
- [ ] [EnlightenGAN](https://github.com/canturan10/light_side/blob/master/light_side/archs/README.md)
- [ ] [MBLLEN](https://github.com/canturan10/light_side/blob/master/light_side/archs/README.md)
- [ ] [LLFlow](https://github.com/canturan10/light_side/blob/master/light_side/archs/README.md)

_For more information, please refer to the [Architectures](https://github.com/canturan10/light_side/blob/master/light_side/archs)_

<!-- DATASETS -->
## Datasets

- [x] [Zero DCE](https://github.com/canturan10/light_side/blob/master/light_side/datasets/README.md)
- [ ] [LOL](https://github.com/canturan10/light_side/blob/master/light_side/datasets/README.md)
- [ ] [DICM](https://github.com/canturan10/light_side/blob/master/light_side/datasets/README.md)
- [ ] [MEF](https://github.com/canturan10/light_side/blob/master/light_side/datasets/README.md)

_For more information, please refer to the [Datasets](https://github.com/canturan10/light_side/blob/master/light_side/datasets)_

<!-- DEPLOYMENTS -->
## Deployments

- [ ] [FastAPI](https://github.com/canturan10/light_side/blob/master/deployment/README.md)
- [ ] [ONNX](https://github.com/canturan10/light_side/blob/master/deployment/README.md)
- [ ] [DeepSparse](https://github.com/canturan10/light_side/blob/master/deployment/README.md)
- [ ] [TensorFlow](https://github.com/canturan10/light_side/blob/master/deployment/README.md)
- [ ] [TensorFlow Lite](https://github.com/canturan10/light_side/blob/master/deployment/README.md)

_For more information, please refer to the [Deployment](https://github.com/canturan10/light_side/blob/master/deployment)_

<!-- TRAINING -->
## Training

To training, follow these steps:

For installing Light Side, please refer to the [Installation](#installation).

```bash
python training/zerodce_training.py
```

For optional arguments,

```bash
python training/zerodce_training.py --help
```

<!-- TESTS -->
## Tests

During development, you might like to have tests run.

Install dependencies

```bash
pip install -e ".[test]"
```

### Linting Tests

```bash
pytest light_side --pylint --pylint-error-types=EF
```

### Document Tests

```bash
pytest light_side --doctest-modules
```

### Coverage Tests

```bash
pytest --doctest-modules --cov light_side --cov-report term
```

<!-- CONTRIBUTING -->
## Contributing

To contribute to `Light Side`, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin`
5. Create the pull request.

Alternatively see the `GitHub` documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

<!-- CONTRIBUTORS -->
## Contributors

<table style="width:100%">
    <tr>
        <td align="center">
            <a href="https://github.com/canturan10">
                <h3>
                    Oğuzcan Turan
                </h3>
                <img src="https://avatars0.githubusercontent.com/u/34894012?s=460&u=722268bba03389384f9d673d3920abacf12a6ea6&v=4&s=200"
                    width="200px;" alt="Oğuzcan Turan" /><br>
                <a href="https://www.linkedin.com/in/canturan10/">
                    <img src="https://img.shields.io/badge/-LinkedIn-blue?style=flat&logo=Linkedin&logoColor=white"
                        width="75px;" alt="Linkedin" />
                </a>
                <a href="https://canturan10.github.io/">
                    <img src="https://img.shields.io/badge/-Portfolio-lightgrey?style=flat&logo=opera&logoColor=white"
                        width="75px;" alt="Portfolio" />
                </a>
            </a>
        </td>
        <td align="center">
            <a href="https://github.com/canturan10">
                <h3>
                    You ?
                </h3>
                <img src="https://raw.githubusercontent.com/canturan10/readme-template/master/src/you.png"
                    width="200px;" alt="Oğuzcan Turan" /><br>
                <a href="#">
                    <img src="https://img.shields.io/badge/-Reserved%20Place-red?style=flat&logoColor=white"
                        width="110px;" alt="Reserved" />
                </a>
            </a>
        </td>
    </tr>
</table>

<!-- CONTACT -->
## Contact

If you want to contact me you can reach me at [can.turan.10@gmail.com](mailto:can.turan.10@gmail.com).

<!-- LICENSE -->
## License

This project is licensed under `MIT` license. See [`LICENSE`](LICENSE) for more information.

<!-- REFERENCES -->
## References

The references used in the development of the project are as follows.

- [Satellighte](https://github.com/canturan10/satellighte)
- [Img Shields](https://shields.io)
- [GitHub Pages](https://pages.github.com)
- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
- [Torchvision](https://github.com/pytorch/vision)

<!-- CITATIONS -->
## Citations

```bibtex
@misc{Turan_satellighte,
author = {Turan, Oguzcan},
title = {{satellighte}},
url = {https://github.com/canturan10/satellighte}
}
```

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

Give a ⭐️ if this project helped you!
![-----------------------------------------------------](https://raw.githubusercontent.com/canturan10/readme-template/master/src/colored_4b.png)

_This readme file is made using the [readme-template](https://github.com/canturan10/readme-template)_
