# colorization-metrics

<!-- markdownlint-disable MD013 -->

[![Python badge](https://img.shields.io/badge/Python-3.13-0066cc?style=for-the-badge&logo=python&logoColor=yellow)](https://www.python.org/downloads/release/python-3135/)

[![uv badge](https://img.shields.io/badge/Packaging-uv-de5ee8?style=for-the-badge&logo=uv)](https://docs.astral.sh/uv/)
[![Ruff badge](https://img.shields.io/badge/Linting/Formatter-Ruff-d6fe64?style=for-the-badge&logo=ruff)](https://docs.astral.sh/ruff/)
[![Pylint badge](https://img.shields.io/badge/Linting-pylint-brightgreen?style=for-the-badge&logo=pylint)](https://pylint.pycqa.org/en/latest/)

> [!NOTE]
> This repository contains the exact peer-reviewed implementation used in our article published in IPOL:
Nicolas Maignan, Fabien Pierre, Frédéric Sur, "A survey of metrics used for image colorization assessment", IPOL, <https://doi.org/10.5201/ipol.2026.632>

> For the up-to-date and maintained version, including potentially new metrics and breaking changes, check the [main branch](https://gitlab.inria.fr/tangram/anr-arce/published-codes/colorization-metrics).

This repository enables image and video colorizations to be evaluated using commonly used metrics, facilitating comparisons with results from the literature.

## TL;DR

```shell
uv run evaluate_colorization -c _colorizationDirectory_ -gt _groundTruthDirectory_
```

## Installation

The project has been configured to use [uv](https://docs.astral.sh/uv/), allowing a virtual environment to be created easily:

```shell
uv sync
```

To use the code located in the `experiments` section, the `dataset` dependency group should be installed:

```shell
uv sync --group dataset
```

The virtual environment can then be activated using the standard command:

```shell
source .venv/bin/activate
```

If preferred, `pip` may be used instead of `uv`, in which case the locked `requirements.txt` file should be used.

## How to use

The main script is `src/colorization_metrics/main_cli.py`. It can be executed as follows:

```shell
evaluate_colorization -c _colorization_ -gt _groundTruth_
```

The arguments `_colorization_` and `_groundTruth_` should be provided as:

- image files, such as `colorized.png` and `gt.png`;
- video files, such as `colorized.mkv` and `gt.mkv`;
- directories containing numbered frame images, such as `colorized/` and `gt/`, where files are named e.g. `img_00024.png` or `007.jpg`.

Additional options can be viewed using the help command:\
`python metric.py -h`

> Note that it can also be executed directly from the file:

```shell
python src/colorization_metrics/main_cli.py -c _colorization_ -gt _groundTruth_
```

## Metrics Implemented

- **PSNR**: The Peak Signal-to-Noise Ratio is calculated in the RGB color space or on the _a_ and _b_ chrominance channels of the CIELAB color space. It is used to assess the similarity between a colorized image and its ground truth. This metric favors colorizations that exactly match the ground truth, possibly penalizing perceptually coherent alternatives.

- **SSIM**: The Structural Similarity Index Measure, introduced in [**Image Quality Assessment: From Error Visibility to Structural Similarity**](https://doi.org/10.1109/TIP.2003.819861) by Wang et al., is used to compare image structure. Although considered more suitable than PSNR or MSE for image data, it has been used less frequently in the literature.

- **LPIPS**: The Learned Perceptual Image Patch Similarity metric, proposed in [**The Unreasonable Effectiveness of Deep Features as a Perceptual Metric**](https://doi.org/10.1109/CVPR.2018.00068) by Zhang et al., is based on deep neural network features. It is applicable to general image processing and is grounded in semantic similarity. No specific evaluation has yet been reported regarding its effectiveness in colorization tasks.

- **FID**: The Fréchet Inception Distance, initially proposed in [**GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium**](https://dl.acm.org/doi/abs/10.5555/3295222.3295408) by Heusel et al., is now used in colorization as well. This metric evaluates perceptual similarity without requiring exact color fidelity but its behavior in colorization contexts remains insufficiently studied.

- **COLORFULNESS**: This perceptual measure, proposed in [**Measuring colourfulness in natural images**](https://doi.org/10.1117/12.477378) by Hasler and Suesstrunk, estimates how colorful an image appears to a human observer. It is derived from linear combinations that are correlated with human judgments.

- **BRISQUE**: The Blind/Referenceless Image Spatial QUality Evaluator, developed in [**No-Reference Image Quality Assessment in the Spatial Domain**](https://doi.org/10.1109/TIP.2012.2214050) by Mittal et al., evaluates images without requiring a reference. Originally limited to grayscale images, it has since been extended to color images, notably in [**Analyse de métriques de qualité pour des images en couleurs**](https://hal.science/hal-05131218v1).

- **NIQE**: The Natural Image Quality Evaluator, introduced in [**Making a “Completely Blind” Image Quality Analyzer**](https://doi.org/10.1109/LSP.2012.2227726) by Mittal et al., compares perceptual features of a target image against a natural image distribution. Like BRISQUE, it was extended to handle color images in later work.

- **MANIQA**: The Multi-dimension Attention-based No-reference Image Quality Assessment, described in [**MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment**](https://doi.org/10.1109/CVPRW56347.2022.00126) by Yang et al., uses transformer-based architectures to predict quality scores. While designed for general image quality, its relevance to colorization remains to be fully validated.

- **CDC**: The Color Distribution Consistency index, described in [**Temporally consistent video colorization with deep feature propagation and self-regularization learning**](https://doi.org/10.48550/arXiv.2110.04562) by Liu et al., evaluates color stability over time in video sequences. It leverages optical flow-based warping.

## Tested on

[![Ubuntu badge](https://img.shields.io/badge/Ubuntu-22.04-cc3300?style=for-the-badge&logo=ubuntu)](https://www.releases.ubuntu.com/22.04/)
[![GPU badge](https://img.shields.io/badge/GPU-RTX_A6000_|_Cuda_12.6-76B900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-12-6-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04)

[![Arch badge](https://img.shields.io/badge/arch-gray?style=for-the-badge&logo=archlinux)](https://archlinux.org/)
[![Intel badge](https://img.shields.io/badge/CPU-%20i7_10510U-blue?style=for-the-badge&logo=intel)](https://ark.intel.com/content/www/fr/fr/ark/products/196449/intel-core-i7-10510u-processor-8m-cache-up-to-4-90-ghz.html)

## Datasets

- **LIVE IQA**: Must be downloaded from the [official website](https://live.ece.utexas.edu/research/Quality/subjective.htm) (Release 2), including the `dmos_realigned.mat` file. These should be placed in `data/live_iqa/`, and then `experiments/scripts/prepare_live_iqa.py` should be executed.

- **Unsplash**: Version 1.2.2 can be obtained from [this link](https://unsplash.com/data/lite/1.2.2). The `.tsv` files should be placed in `data/unsplash/`.

- **DAVIS (dance-twirl)**: This video is part of the [DAVIS 2017 Semi-supervised Validation Dataset](https://davischallenge.org/davis2017/code.html#semisupervised) and should be placed in the `data/video/` directory.

## Acknowledgments

- FID is computed using PyTorch via the [pytorch-fid](https://github.com/mseitzer/pytorch-fid/tree/master) module, as recommended by the original authors.

- LPIPS is computed using the official [lpips](https://github.com/richzhang/PerceptualSimilarity) implementation.

- BRISQUE and NIQE use a Python reimplementation of MATLAB’s `imresize`, provided by Aleksandr Petiushko ([GitHub](https://github.com/fatheral/matlab_imresize)), and are integrated via the [matpy](https://gitlab.com/nifra/matpy) module.

- MANIQA has been converted into a standalone Python library based on the [original implementation](https://github.com/IIGROUP/MANIQA), and is available via [this repository](https://gitlab.inria.fr/nmaignan/MANIQA).
