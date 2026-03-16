# Datasets Information

This package includes three datasets for plant part segmentation:

## 1. Pheno4D-maize (Included)

**Location**: `./data/`

**Description**: Maize plant point clouds with part-level annotations

**Statistics**:
- 7 plant categories (M01-M07)
- 28 part classes (stems, leaves, etc.)
- Point cloud size: 65K - 2.1M points per sample

**Original Dataset**: Pheno4D
- Website: https://www.ipb.uni-bonn.de/data/pheno4d/index.html

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

---

## 2. Crops3D (Download Link)

**Download**: https://github.com/clawCa/Crops3D

**Description**: Diverse 3D crop dataset for agricultural applications

**Statistics**:
- 8 crop categories
- 27 part classes
- Multiple growth stages and crop types

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

## 3. ShapeNet (Download Link)

**Download**: https://pan.baidu.com/s/1wcu6zOvKNyNv6FZ13-KSvA
**Access Code**: `ikxu`

**Description**: Large-scale 3D shape dataset with part annotations

**Statistics**:
- 16 object categories
- 50 part classes
- Diverse 3D shapes for part segmentation

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

| Dataset | Categories | Part Classes | Status |
|---------|-----------|--------------|--------|
| Pheno4D-maize | 7 | 28 | ✅ Included in `./data/` |
| Crops3D | 8 | 27 | 🔗 Download link provided |
| ShapeNet | 16 | 50 | 🔗 Download link provided |

---

## Data Format

All datasets use the same point cloud format:

```
x y z label
```

- `x, y, z`: 3D coordinates (float)
- `label`: Part label index (integer)

Example:
```
0.123 0.456 0.789 5
0.234 0.567 0.890 5
0.345 0.678 0.901 12
...
```

---

## Usage

To test the model on different datasets:

```bash
# Pheno4D-maize (included)
python test_model.py --input data/sample.txt --output result.txt

# Crops3D (after download)
python test_model.py --input crops3d/sample.txt --output result.txt

# ShapeNet (after download)
python test_model.py --input shapenet/sample.txt --output result.txt
```
