import numpy as np
import tensorflow as tf
import os
import random
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.model import build_mt_masknet
from src.utils import load_phase2_data, plot_cm, plot_roc, save_roc_csv, write_phase2_report

def phase2_train_mt_masknet(config, seed, results_path):
    """Train MT-MaskNet for a single seed (phase 2).
    
    Args:
        config: Dict from YAML.
        seed: Random seed.
        results_path: Path to append report.
    
    Returns:
        voice_test_acc: Float for mean±std (from voice report).
    """
    # Set seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Load pre-trained paths
    pretrained_dir = config['phase2']['pretrained_models_dir']
    sound_path = os.path.join(pretrained_dir, f'sound_seed{seed}.keras')
    noise_path = os.path.join(pretrained_dir, f'noise_seed{seed}.keras')
    
    # Build model
    model = build_mt_masknet(sound_path, noise_path)
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['phase2']['learning_rate']),
        loss={'voice_type_output': 'categorical_crossentropy',
              'noise_type_output': 'categorical_crossentropy'},
        loss_weights={'voice_type_output': config['phase2']['loss_weights']['voice_type'], 
                      'noise_type_output': config['phase2']['loss_weights']['noise_type']},
        metrics={'voice_type_output': 'accuracy', 'noise_type_output': 'accuracy'}
    )
    
    # Load data
    train_data, val_data, test_data = load_phase2_data(config)
    X_train_spec, Y_bio_train, Y_noise_train = train_data
    X_val_spec, Y_bio_val, Y_noise_val = val_data
    X_test_spec, Y_bio_test, Y_noise_test = test_data
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_voice_type_output_accuracy', mode='max', 
                                   patience=config['phase2']['early_stop_patience'], 
                                   restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_voice_type_output_accuracy', 
                                  factor=config['phase2']['reduce_lr_factor'], 
                                  patience=config['phase2']['reduce_lr_patience'], 
                                  min_lr=config['phase2']['min_lr'], mode='max', verbose=1)
    
    # Train
    history = model.fit(
        [X_train_spec, X_train_spec],
        {'voice_type_output': Y_bio_train, 'noise_type_output': Y_noise_train},
        epochs=config['phase2']['epochs'],
        batch_size=config['phase2']['batch_size'],
        validation_data=([X_val_spec, X_val_spec], {'voice_type_output': Y_bio_val, 'noise_type_output': Y_noise_val}),
        callbacks=[reduce_lr, early_stopping],
        verbose=1
    )
    
    # Get best metrics (focus on voice as primary task)
    best_epoch = early_stopping.stopped_epoch if early_stopping.stopped_epoch > 0 else len(history.history['loss'])
    best_val_acc = max(history.history['val_voice_type_output_accuracy'])
    
    # Save MT-MaskNet model
    mt_models_dir = config['phase2']['mt_models_dir']
    os.makedirs(mt_models_dir, exist_ok=True)
    model_path = os.path.join(mt_models_dir, f'mt_masknet_seed{seed}.keras')
    model.save(model_path)
    print(f"Saved MT-MaskNet model for seed {seed}: {model_path}")
    
    # Evaluate on test
    Y_pred = model.predict([X_test_spec, X_test_spec])
    Y_pred_voice = np.argmax(Y_pred[0], axis=1)
    Y_pred_noise = np.argmax(Y_pred[1], axis=1)
    Y_true_voice = np.argmax(Y_bio_test, axis=1)
    Y_true_noise = np.argmax(Y_noise_test, axis=1)
    
    # Reports
    voice_report = classification_report(Y_true_voice, Y_pred_voice, digits=4)
    noise_report = classification_report(Y_true_noise, Y_pred_noise, digits=4)
    
    # Plots and CSV
    figures_dir = config['phase2']['figures_dir']
    os.makedirs(figures_dir, exist_ok=True)
    cm_voice_path = os.path.join(figures_dir, f'mt_seed{seed}_voice_cm.png')
    cm_noise_path = os.path.join(figures_dir, f'mt_seed{seed}_noise_cm.png')
    roc_plot_path = os.path.join(figures_dir, f'mt_seed{seed}_voice_roc.png')
    roc_csv_path = os.path.join(figures_dir, f'mt_seed{seed}_voice_roc.csv')
    
    plot_cm(Y_true_voice, Y_pred_voice, cm_voice_path, f'VoiceType CM (Seed={seed})')
    plot_cm(Y_true_noise, Y_pred_noise, cm_noise_path, f'NoiseType CM (Seed={seed})')
    plot_roc(Y_true_voice, Y_pred[0], 3, roc_plot_path)  # Voice has 3 classes
    save_roc_csv(Y_true_voice, Y_pred[0], roc_csv_path)
    
    # Write report
    write_phase2_report(results_path, seed, best_epoch, best_val_acc, 
                        voice_report, noise_report, cm_voice_path, cm_noise_path, 
                        roc_plot_path, roc_csv_path)
    
    # Extract voice test acc for mean±std
    voice_test_acc = float(voice_report.split('accuracy')[1].split()[0])
    return voice_test_acc