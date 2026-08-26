"""Compute various image and video quality metrics.

This module aggregates several metrics for evaluating the quality of colorized images
and videos. It includes functions to compute :
- [Blind/Referenceless Image Spatial QUality Evaluator (BRISQUE)](https://doi.org/10.1109/TIP.2012.2214050)
- [Temporally Consistent Video Colorfulness (CDC)](https://doi.org/10.48550/arXiv.2110.04562),
- [colorfulness metric](https://doi.org/10.1117/12.477378),
- [Fréchet Inception Distance (FID)](https://doi.org/10.48550/arXiv.1706.08500),
- [Learned Perceptual Image Patch Similarity (LPIPS)](https://doi.org/10.1109/CVPR.2018.00068),
- [Multi-dimension Attention Network for No-Reference Image Quality Assessment (MANIQA)](https://doi.org/10.1109/CVPRW56347.2022.00126)
- [Natural Image Quality Evaluator (NIQE)](https://doi.org/10.1109/LSP.2012.2227726)
- Peak Signal-to-Noise Ratio (PSNR),
- [Structural Similarity Index (SSIM)](https://doi.org/10.1109/TIP.2003.819861).
"""
