# Crop-PVCNet-main
# Pheno4D-maize: Plant Part Segmentation Model (Submission Package)

> **Note**: This is a minimal package for paper review. Complete code will be released upon acceptance.

## Overview

Trained model weights and inference code for plant part segmentation on multiple 3D plant point cloud datasets.

**Performance**: mIoU 98.51%, Accuracy 99.3% (on Pheno4D-maize)
**Model**: PVT/PVCT (Point-Voxel-Transformer)
**Datasets**: Pheno4D-maize, Crops3D, ShapeNet

---

## Quick Start

### Requirements
```bash
pip install torch>=1.8.0 numpy scipy tqdm
```

### Test Model
```python
python test_model.py --input sample.txt --output result.txt
```

### Input Format
```
x y z label
```
- `x, y, z`: 3D coordinates
- `label`: ground truth part label (0-27)

---

## Package Contents

```
├── checkpoints/
│   └── partmodel.t7          # Trained model (14MB, pure weights)
├── model/                     # Core model definitions
│   ├── partpvt.py
│   └── partpvct.py
├── modules/                   # Essential modules
│   ├── pvtconv.py
│   ├── pvctconv.py
│   └── ...
├── data/                      # Pheno4D-maize dataset
├── test_model.py             # Inference script
└── README.md                 # This file
```

---

## Model Architecture

**PVT/PVCT** combines:
- Point-based processing
- Voxel-based 3D CNN
- Transformer attention

---

## Datasets

### 1. Pheno4D-maize (Provided)

**Description**: Maize plant point clouds with part-level annotations
**Statistics**:
- 7 plant categories (M01-M07)
- 28 part classes (stems, leaves, etc.)
- Point cloud size: 65K - 2.1M points per sample

**Download**: Included in `/home/lj/gaoyijie/PVT-main/data/`

**Original Dataset**: Pheno4D
**Citation**:
```bibtex
@article{schunck2021pheno4d,
  title={Pheno4D: A spatio-temporal dataset of maize and tomato plant point clouds for phenotyping and advanced plant analysis},
  author={Schunck, D. and Magistri, F. and Rosu, R. A. and Corneli{\ss}en, A. and Chebrolu, N. and Paulus, S. and L{\'e}on, J. and Behnke, S. and Stachniss, C. and Kuhlmann, H. and Klingbeil, L.},
  journal={PLOS ONE},
  volume={16},
  number={8},
  pages={1--18},
  year={2021},
  doi={10.1371/journal.pone.0256340}
}
```
**Original Download**: https://www.ipb.uni-bonn.de/data/pheno4d/index.html

---

### 2. Crops3D (Download Link Provided)

**Description**: Diverse 3D crop dataset for agricultural applications
**Statistics**:
- 8 crop categories
- 27 part classes
- Multiple growth stages and crop types

**Download**: https://github.com/clawCa/Crops3D

**Citation**:
```bibtex
@article{crops3d2024,
  title={Crops3D: A Diverse 3D Crop Dataset for Realistic Perception and Segmentation toward Agricultural Applications},
  author={Zhu, J. and Zhai, R. and Ren, H. and others},
  journal={Scientific Data},
  volume={11},
  number={1438},
  year={2024},
  doi={10.1038/s41597-024-04290-0}
}
```

---

### 3. ShapeNet (Download Link Provided)

**Description**: Large-scale 3D shape dataset with part annotations
**Statistics**:
- 16 object categories
- 50 part classes
- Diverse 3D shapes for part segmentation

**Download**: https://pan.baidu.com/s/1wcu6zOvKNyNv6FZ13-KSvA
**Access Code**: `ikxu`

**Citation**:
```bibtex
@article{yi2016scalable,
  title={A scalable active framework for region annotation in 3d shape collections},
  author={Yi, Li and Kim, Vladimir G and Ceylan, Duygu and Shen, I-Chao and Yan, Mengyan and Su, Hao and Lu, Cewu and Huang, Qixing and Sheffer, Alla and Guibas, Leonidas},
  journal={ACM Transactions on Graphics (TOG)},
  volume={35},
  number={6},
  pages={1--12},
  year={2016},
  publisher={ACM New York, NY, USA}
}
```

---

## Dataset Summary

| Dataset | Categories | Part Classes | Provided |
|---------|-----------|--------------|----------|
| Pheno4D-maize | 7 | 28 | ✅ Included |
| Crops3D | 8 | 27 | 🔗 Link provided |
| ShapeNet | 16 | 50 | 🔗 Link provided |

---

## Citation

If you use this model or datasets in your research, please cite:

**Our Work**:
```bibtex
@article{pheno4d_maize_segmentation_2026,
  title={Pheno4D-maize: Large-scale 3D Plant Point Cloud Segmentation with PVT/PVCT},
  author={Your Name},
  journal={Under Review},
  year={2026}
}
```

**Pheno4D Dataset**:
```bibtex
@article{schunck2021pheno4d,
  title={Pheno4D: A spatio-temporal dataset of maize and tomato plant point clouds for phenotyping and advanced plant analysis},
  author={Schunck, D. and Magistri, F. and Rosu, R. A. and Corneli{\ss}en, A. and Chebrolu, N. and Paulus, S. and L{\'e}on, J. and Behnke, S. and Stachniss, C. and Kuhlmann, H. and Klingbeil, L.},
  journal={PLOS ONE},
  volume={16},
  number={8},
  pages={1--18},
  year={2021},
  doi={10.1371/journal.pone.0256340}
}
```

**Crops3D Dataset**:
```bibtex
@article{crops3d2024,
  title={Crops3D: A Diverse 3D Crop Dataset for Realistic Perception and Segmentation toward Agricultural Applications},
  author={Zhu, J. and Zhai, R. and Ren, H. and others},
  journal={Scientific Data},
  volume={11},
  number={1438},
  year={2024},
  doi={10.1038/s41597-024-04290-0}
}
```

**ShapeNet Dataset**:
```bibtex
@article{yi2016scalable,
  title={A scalable active framework for region annotation in 3d shape collections},
  author={Yi, Li and Kim, Vladimir G and Ceylan, Duygu and Shen, I-Chao and Yan, Mengyan and Su, Hao and Lu, Cewu and Huang, Qixing and Sheffer, Alla and Guibas, Leonidas},
  journal={ACM Transactions on Graphics (TOG)},
  volume={35},
  number={6},
  pages={1--12},
  year={2016},
  publisher={ACM New York, NY, USA}
}
```

---

## Notes

- This package contains **inference code only** for review purposes
- Training code, data preprocessing, and CUDA implementations are **not included**
- **Complete code** will be publicly released upon paper acceptance
- For questions during review, please contact: [liujie@htu.edu.cn]

---

## License

MIT License - See LICENSE file for details

