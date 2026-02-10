# 📁 Project Structure

## Directory Organization

```
fc-aml-genai/
├── 📁 models/              # Trained model checkpoints (.pth)
│   ├── beta_vae_final.pth           (102MB - Main VAE model)
│   ├── best_beta_vae.pth             (Best VAE checkpoint)
│   ├── beta_vae_epoch_*.pth          (Training checkpoints)
│   ├── disorder_classifier_best.pth  (Best classifier)
│   ├── disorder_classifier_final.pth (Final classifier)
│   └── latent_representations.pth    (Extracted latents)
│
├── 📁 visualizations/      # All plots and figures (.png)
│   ├── training_curves.png               (β-VAE training)
│   ├── classifier_training_curves.png    (Classifier training)
│   ├── confusion_matrix.png              (Classification heatmap)
│   ├── tsne_latent_space.png            (Latent clusters)
│   ├── reconstruction_examples.png       (VAE quality)
│   ├── reconstruction_quality.png        (Detailed VAE analysis)
│   ├── latent_interpolations.png         (Disorder transitions)
│   ├── latent_traversal.png             (Dimension exploration)
│   ├── visual_evaluation.png             (Classification grid)
│   └── prediction_*.png                  (Individual predictions)
│
├── 📁 data/                # Datasets and annotations (.csv)
│   ├── train.csv                  (Original training set)
│   ├── train_augmented.csv        (With synthetic faces)
│   ├── val.csv                    (Validation set)
│   ├── test.csv                   (Test set)
│   ├── annotations_balanced.csv   (Balanced dataset)
│   ├── synthetic_annotations.csv  (StyleGAN2 faces)
│   └── features_raw.csv           (Feature extraction)
│
├── 📁 outputs/             # Text reports and logs (.txt)
│   ├── classification_report.txt  (Per-class metrics)
│   ├── training_classifier.txt    (Classifier logs)
│   └── eval_results.txt           (Evaluation results)
│
├── 📁 scripts/             # (Reserved for helper scripts)
│
├── 📁 ffhq_images/         # Face images (512×512 PNG)
├── 📁 synthetic_faces/     # StyleGAN2 generated faces
├── 📁 stylegan2-ada-pytorch/  # StyleGAN2 repository
├── 📁 cascades/            # OpenCV Haar cascade files
│
├── 🐍 train_beta_vae.py             # VAE training
├── 🐍 classifier_training.py        # Classifier training
├── 🐍 evaluate_model.py             # Complete evaluation
├── 🐍 visual_evaluation.py          # Grid visualization
├── 🐍 reconstruction_visual.py      # VAE reconstruction analysis
├── 🐍 predict_disorder.py           # Live inference
├── 🐍 latent_interpolation.py       # Disorder transitions
├── 🐍 latent_traversal.py           # Dimension exploration
├── 🐍 generate_synthetic_faces.py   # StyleGAN2 generation
├── 🐍 annotate_synthetic.py         # Synthetic annotation
├── 🐍 merge_datasets.py             # Dataset merging
├── 🐍 new_auto_annotation.py        # Feature extraction
└── 📄 README.md                     # Project documentation
```

## Quick Reference

### Training Pipeline
```bash
# 1. Train β-VAE (outputs to models/)
python train_beta_vae.py

# 2. Train classifier (outputs to models/)
python classifier_training.py

# 3. Evaluate (outputs to visualizations/ and outputs/)
python evaluate_model.py
```

### Visualization Scripts
```bash
# Grid of predictions with color coding
python visual_evaluation.py

# Detailed reconstruction analysis
python reconstruction_visual.py

# Disorder transition animations
python latent_interpolation.py

# Dimension exploration
python latent_traversal.py
```

### Inference
```bash
# Predict on single image
python predict_disorder.py --image ffhq_images/00001.png --show-viz
```

## File Size Summary

- **Models**: ~520MB total (10 .pth files)
- **Visualizations**: ~15MB total (10 .png files)
- **Data**: ~2MB total (9 .csv files)
- **Images**: ~350MB (ffhq_images + synthetic_faces)

## Notes

- All scripts automatically output to organized directories
- Models are saved incrementally during training
- Visualizations include publication-ready figures
- CSV files include both original and augmented datasets
