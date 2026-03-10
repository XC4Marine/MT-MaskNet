# MT-MaskNet: Noise-Masking Marine Mammal Sound Classification

Official implementation of the paper: **A Noise-Masking Marine Mammal Sound Classification Method via Multi-Task Learning** (2026).

This repository provides modular code for training, testing, and ablation studies on the MT-MaskNet model, an end-to-end multi-task learning framework for marine mammal sound classification under noisy conditions.


# Phase 1: Pre-training

## Setup
1. Install: `pip install -r requirements.txt`
2. Data: Place .npy files in `data/{Train/Val/Test}_feature/`
3. Run: `python scripts/run_experiments.py`

This runs 10 seeds for sound and noise pre-training, generates CM/ROC plots, classification reports, and final mean±std in `results/reports/phase1_results.txt`.

## Expected Output
- Training structure: See ResNet18 summary printed at end.
- Per-seed reports: In TXT (example: precision/recall/F1 per class, accuracy).
- Mean ± STD: Calculated from test accuracies (e.g., Sound: 0.9500 ± 0.0080).

## Phase 2: MT-MaskNet Training
1. Ensure phase 1 is run (pre-trained .keras in results/models/).
2. Run: `python scripts/phase2_run_experiments.py`
   - By default, runs only first seed (to verify setup/layer names).
   - To run all 10: Uncomment the for loop in phase2_run_experiments.py.
3. Layer Verification: Before full run, load a phase1 model (e.g., in Python shell):

import tensorflow as tf
model = tf.keras.models.load_model('results/models/sound_seed100.keras')
model.summary()  # Check ReLU/BatchNorm names; update src/model.py if needed (e.g., re_lu_6 -> re_lu_5)

This ensures gating layers match your env.

Outputs: MT-MaskNet .keras per seed, CM/ROC plots/CSV, reports with mean±std in results/reports/phase2_results.txt.

## Real Data Testing
1. Place .wav files in `data/realPAM/` (WhiteBeaked.wav, WhiteSided.wav, AtlanticSpotted.wav).
2. Run: `python scripts/test_on_real.py --model_type sound --seed 100` (or mt_masknet for phase2 model).
   - Outputs only classification report to console.
3. If STFT shape != 128x128, update generate_stft_spectrograms() with resize (e.g., import scipy.ndimage.zoom).

## Ablation Study on Gating Layers
This is an example for gating at Layer 1 (based on your notebook). To study other layers:
1. Load a phase1 model: `model = tf.keras.models.load_model('results/models/sound_seed100.keras'); model.summary()` — note ReLU/BN names for the target layer.
2. Update config.yaml: Set `ablation.layer_relu` and `ablation.layer_bn` to those names.
3. Run: `python scripts/ablation/ablation_layer1.py` (duplicate/rename script for other layers if needed).
- Outputs classification reports + mean±std to results/reports/ablation_results_layer1.txt for original test and real PAM sets.

## Data Sources

This repository uses pre-processed datasets for training and testing MT-MaskNet. All data is stored in the `data/` directory after preprocessing (e.g., .npy files for spectrograms). Raw sources are public and cited below. Ensure you comply with their usage terms (e.g., attribution).

- **Marine Mammal Sounds (Dolphin Species)**: Sourced from the Watkins Marine Mammal Sound Database, providing recordings of three dolphin species (e.g., White-beaked, White-sided, Atlantic spotted).  
  Website: [Watkins Collection](https://cis.whoi.edu/science/B/whalesounds/index.cfm)  


- **Noise Types (Wind, Rain, Shipping, Shrimp)**: Realistic underwater noise from the SanctSound project (NOAA and partners), used for augmentation at multiple SNRs.  
  Website: [SanctSound Portal](https://sanctsound.ioos.us/)  


- **Real-World PAM Recordings**: For the real-world PAM test set, we additionally prepared a collection of long-duration field recordings:
1.one 50s clip of White-sided Dolphin from Ocean Conservation Research,
2.one 50s of Atlantic-Spotted Dolphin from the Watkins database (excluded from the ‘Best Cut’ training subset),
3.one 26s clip of White-beaked Dolphin from North Sailing.