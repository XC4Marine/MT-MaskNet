import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.model import build_resnet18
from src.utils import load_data, plot_cm, plot_roc, write_report
import random
import os

def train_sound(config, seed, results_path):
    """Train ResNet18 for sound type classification (single seed).
    
    Args:
        config: Dict from YAML.
        seed: Random seed.
        results_path: Path to append report.
    
    Returns:
        test_accuracy: Float for mean±std calculation.
    """
    # Set seeds for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Load data
    train_data, val_data, test_data = load_data(config, 'sound')
    X_train, Y_train = train_data
    X_val, Y_val = val_data
    X_test, Y_test = test_data
    
    # Build and compile model
    model = build_resnet18(config['training']['input_shape'], config['training']['sound_classes'])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=config['training']['early_stop_patience'], 
                                   restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=config['training']['reduce_lr_factor'], 
                                  patience=config['training']['reduce_lr_patience'], min_lr=config['training']['min_lr'], 
                                  mode='max', verbose=1)
    
    # Train
    history = model.fit(X_train, Y_train, epochs=config['training']['epochs'], batch_size=config['training']['batch_size'],
                        validation_data=(X_val, Y_val), callbacks=[reduce_lr, early_stopping], verbose=1)
    # Save the best model
    models_dir = config['paths']['models_dir']
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f'sound_seed{seed}.keras')
    model.save(model_path)
    print(f"Saved sound model for seed {seed}: {model_path}")

    # Get best metrics
    best_epoch = early_stopping.stopped_epoch if early_stopping.stopped_epoch > 0 else len(history.history['loss'])
    best_val_acc = max(history.history['val_accuracy'])
    
    # Evaluate on test
    Y_pred_prob = model.predict(X_test)
    Y_pred = np.argmax(Y_pred_prob, axis=1)
    Y_true = np.argmax(Y_test, axis=1)
    
    # Report and plots
    report = classification_report(Y_true, Y_pred, digits=4)
    figures_dir = config['paths']['figures_dir']
    os.makedirs(figures_dir, exist_ok=True)
    cm_path = os.path.join(figures_dir, f'sound_seed{seed}_cm.png')
    roc_path = os.path.join(figures_dir, f'sound_seed{seed}_roc.png')
    plot_cm(Y_true, Y_pred, cm_path, f'Sound CM (Seed {seed})')
    plot_roc(Y_true, Y_pred_prob, config['training']['sound_classes'], roc_path)
    
    write_report(results_path, seed, best_epoch, best_val_acc, report, cm_path, roc_path)
    
    test_acc = float(report.split('accuracy')[1].split()[0])  # Extract acc from report
    return test_acc