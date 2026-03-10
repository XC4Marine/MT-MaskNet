import numpy as np
import tensorflow as tf
import os
import yaml
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
from src.model import build_mt_masknet
from src.utils import load_phase2_data, load_and_segment_real_data, generate_stft_spectrograms, write_phase2_report  # Reuse/adapt

def ablation_layer1(config_path):
    """Example ablation study for gating at Layer 1.
    
    Trains/evaluates MT-MaskNet with Layer1 gating on 5 seeds.
    Outputs classification reports for original test and real PAM sets.
    
    To adapt for other layers:
    1. Run model.summary() on a loaded phase1 model to find ReLU/BN names.
    2. Update config['ablation']['layer_relu'] and ['layer_bn'].
    3. Rerun or duplicate script for that layer.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    results_path = config['ablation']['results_txt']
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    voice_accs_original = []
    voice_accs_real = []
    seeds = [100, 200, 300, 400, 500,600, 700, 800, 900, 1000]  # As in notebook
    
    for seed in seeds:
        # Set seeds
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Load pre-trained
        pretrained_dir = config['phase2']['pretrained_models_dir']  # Reuse phase2 config
        sound_path = os.path.join(pretrained_dir, f'sound_seed{seed}.keras')
        noise_path = os.path.join(pretrained_dir, f'noise_seed{seed}.keras')
        
        # Build with ablation layers
        model = build_mt_masknet(sound_path, noise_path, 
                                 config['ablation']['layer_relu'], 
                                 config['ablation']['layer_bn'])
        
        # Compile (reuse phase2 params)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config['phase2']['learning_rate']),
            loss={'voice_type_output': 'categorical_crossentropy', 'noise_type_output': 'categorical_crossentropy'},
            loss_weights={'voice_type_output': 1.0, 'noise_type_output': 1.0},  # As in ablation notebook (1:1)
            metrics={'voice_type_output': 'accuracy', 'noise_type_output': 'accuracy'}
        )
        
        # Load original data
        train_data, val_data, test_data = load_phase2_data(config)
        X_train_spec, Y_bio_train, Y_noise_train = train_data
        X_val_spec, Y_bio_val, Y_noise_val = val_data
        X_test_spec, Y_bio_test, Y_noise_test = test_data
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_voice_type_output_accuracy', mode='max', 
                                       patience=config['phase2']['early_stop_patience'], restore_best_weights=True, verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor='val_voice_type_output_accuracy', factor=0.5, 
                                      patience=3, min_lr=1e-6, mode='max', verbose=0)
        
        # Train (silent)
        model.fit([X_train_spec, X_train_spec], {'voice_type_output': Y_bio_train, 'noise_type_output': Y_noise_train},
                  epochs=100, batch_size=128, validation_data=([X_val_spec, X_val_spec], {'voice_type_output': Y_bio_val, 'noise_type_output': Y_noise_val}),
                  callbacks=[reduce_lr, early_stopping], verbose=0)
        
        # Save ablation model
        models_dir = config['ablation']['models_dir']
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f'mt_masknet_layer1_seed{seed}.keras')
        model.save(model_path)
        
        # Evaluate on original test set
        Y_pred = model.predict([X_test_spec, X_test_spec], verbose=0)
        Y_pred_voice = np.argmax(Y_pred[0], axis=1)
        Y_true_voice = np.argmax(Y_bio_test, axis=1)
        voice_report_original = classification_report(Y_true_voice, Y_pred_voice, digits=4)
        voice_accs_original.append(float(voice_report_original.split('accuracy')[1].split()[0]))
        
        # Evaluate on real PAM
        X_segments_real, y_true_real = load_and_segment_real_data(config)
        X_specs_real = generate_stft_spectrograms(X_segments_real, config['real_test']['sr'])
        Y_pred_real = model.predict([X_specs_real, X_specs_real], verbose=0)
        Y_pred_voice_real = np.argmax(Y_pred_real[0], axis=1)
        voice_report_real = classification_report(y_true_real, Y_pred_voice_real, digits=4)
        voice_accs_real.append(float(voice_report_real.split('accuracy')[1].split()[0]))
        
        # Write reports (minimal)
        with open(results_path, 'a') as f:
            f.write(f"\nSeed {seed} - Original Test Report (Voice):\n{voice_report_original}\n")
            f.write(f"Seed {seed} - Real PAM Report (Voice):\n{voice_report_real}\n")
    
    # Summary mean±std
    with open(results_path, 'a') as f:
        f.write("\nSummary (Voice Accuracy):\n")
        f.write(f"Original Test: {np.mean(voice_accs_original):.4f} ± {np.std(voice_accs_original):.4f}\n")
        f.write(f"Real PAM: {np.mean(voice_accs_real):.4f} ± {np.std(voice_accs_real):.4f}\n")

if __name__ == "__main__":
    ablation_layer1('configs/config.yaml')