# Facial Anomaly Detection with β-VAE and StyleGAN2

**Intégration de β-VAE et StyleGAN2 pour la Génération et l'Analyse de Variations Faciales dans l'Identification de Pathologies**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

End-to-end pipeline for detecting facial anomalies using hybrid generative models. Combines β-VAE for disentangled representation learning with StyleGAN2 for data augmentation.

##  Key Results

- **56.67% accuracy** on 6-class facial anomaly classification (3.4× better than random)
- **93.51% F1-score** for facial asymmetry detection
- **2,846 training samples** (2,100 real + 746 synthetic via StyleGAN2)
- **128-D latent space** enabling controlled facial variation generation

##  Project Structure

```
fc-aml-genai/
├──  models/              # Trained model checkpoints (.pth)
├──  visualizations/      # Generated plots and figures (.png)
├──  data/                # Datasets and annotations (.csv)
├──  outputs/             # Reports and logs (.txt)
├──  ffhq_images/         # Original face images (512×512)
├──  synthetic_faces/     # StyleGAN2 generated faces
│
├──  train_beta_vae.py             # β-VAE training
├──  classifier_training.py        # Classifier training
├──  evaluate_model.py             # Complete evaluation
├──  visual_evaluation.py          # Grid visualization
├──  reconstruction_visual.py      # VAE reconstruction analysis
├──  reconstruction_individual.py  # Per-class reconstructions
├──  predict_disorder.py           # Live inference
├──  latent_interpolation.py       # Disorder transitions
├──  latent_traversal.py           # Dimension exploration
│
├──  generate_synthetic_faces.py   # StyleGAN2 generation
├──  annotate_synthetic.py         # Synthetic annotation
├──  merge_datasets.py             # Dataset merging
├──  new_auto_annotation.py        # Feature extraction
│
├──  report.tex                    # LaTeX report (French)
├──  references.bib                # Bibliography
└──  README.md                     # This file
```

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd fc-aml-genai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn scikit-learn pillow opencv-python tqdm click
```

### 2. Dataset Setup

Place FFHQ images in `ffhq_images/` and run annotation:

```bash
python new_auto_annotation.py
```

This generates:
- `data/train.csv`, `data/val.csv`, `data/test.csv` (balanced splits)
- Automatic facial feature extraction and labeling

### 3. Training Pipeline

```bash
# Step 1: Train β-VAE (18 minutes on RTX 4070)
python train_beta_vae.py

# Step 2: Train classifier (10 minutes)
python classifier_training.py

# Step 3: Evaluate
python evaluate_model.py
```

**Output:**
- Models saved to `models/`
- Training curves to `visualizations/`
- Classification report to `outputs/`

##  Evaluation & Visualization

### Complete Evaluation

```bash
python evaluate_model.py
```

Generates:
- `confusion_matrix.png` - 6×6 heatmap
- `tsne_latent_space.png` - 2D projection
- `reconstruction_examples.png` - VAE quality
- `classification_report.txt` - Per-class metrics

### Enhanced Visualizations

```bash
# Grid of predictions with color-coded results
python visual_evaluation.py --samples-per-class 8

# Detailed reconstruction analysis
python reconstruction_individual.py --samples-per-class 4

# Latent space analysis
python latent_interpolation.py
python latent_traversal.py
```

### Live Inference

```bash
python predict_disorder.py --image ffhq_images/00001.png --show-viz
```

##  Data Augmentation (Optional)

Generate synthetic faces with StyleGAN2:

```bash
# 1. Generate faces
python generate_synthetic_faces.py --num-samples 3500

# 2. Annotate
python annotate_synthetic.py

# 3. Merge with original data
python merge_datasets.py

# 4. Retrain models on augmented data
python train_beta_vae.py
python classifier_training.py
```

**Note:** Requires StyleGAN2-ADA-PyTorch. See `stylegan_setup.py` for setup.

## Results Summary

### Classification Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Facial Asymmetry | 91.14% | 96.00% | **93.51%** |
| Short Lower Face | 64.10% | 66.67% | 65.36% |
| Hypotelorism | 54.05% | 53.33% | 53.69% |
| Long Lower Face | 47.06% | 53.33% | 50.00% |
| Normal | 45.76% | 36.00% | 40.30% |
| Hypertelorism | 34.67% | 34.67% | 34.67% |
| **Overall** | **56.13%** | **56.67%** | **56.25%** |

### Key Findings

 **Strong asymmetry detection** - Binary features (symmetric/asymmetric) well captured  
 **Challenging eye spacing** - Continuous features harder to classify  
 **Modest augmentation gain** - +0.89% due to synthetic label ceiling  
 **Latent space structure** - Enables controlled facial variation generation  

## LaTeX Report

Full academic report in French following professor's structure:

```bash
# Compile report
cd /mnt/d/fc-aml-genai
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

**Or use Overleaf:**
1. Upload `report.tex`, `references.bib`
2. Create `visualizations/` folder
3. Upload all 8 PNG figures
4. Compile automatically

**Report includes:**
- Complete methodology (β-VAE + StyleGAN2 architecture)
- All results with 8 publication-quality figures
- Limitations and future work
- 10 academic references

##  Hardware Requirements

**Minimum:**
- GPU: 6GB VRAM (GTX 1060 or better)
- RAM: 16GB
- Storage: 10GB

**Tested on:**
- NVIDIA RTX 4070 Laptop GPU (8GB VRAM)
- Training time: ~30 minutes total (β-VAE + classifier)

##  Key Dependencies

- Python 3.8+
- PyTorch 2.0+
- torchvision
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- OpenCV (cv2)
- PIL (Pillow)

##  Citation

If you use this code for your research, please cite:

```bibtex
@software{facial_anomaly_detection_2026,
  title={Facial Anomaly Detection with β-VAE and StyleGAN2},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/fc-aml-genai}
}
```

##  Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

##  License

This project is licensed under the MIT License - see LICENSE file for details.

##  References

- Higgins et al. (2017) - β-VAE for disentangled representations
- Karras et al. (2020) - StyleGAN2 for face generation
- Ferry et al. (2014) - Facial dysmorphology detection
- Gurovich et al. (2019) - Deep learning for genetic disorders

##  Contact

For questions or collaborations, please open an issue or reach out via email.

---

**Status:**  Complete - Ready for academic submission and deployment

**Last Updated:** February 2026
