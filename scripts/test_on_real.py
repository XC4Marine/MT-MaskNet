import numpy as np
import tensorflow as tf
import yaml
import argparse
from sklearn.metrics import classification_report
from src.utils import load_and_segment_real_data, generate_stft_spectrograms

def test_on_real(config_path, model_type='sound', seed=100):
    """Test a loaded model on real PAM data and output only classification report.
    
    Args:
        config_path: Path to YAML config.
        model_type: 'sound' (phase1 ResNet18) or 'mt_masknet' (phase2).
        seed: Seed for the model to load.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed for reproducibility
    np.random.seed(config['real_test']['seed'])
    tf.random.set_seed(config['real_test']['seed'])
    
    # Load and segment real data
    X_segments, y_true = load_and_segment_real_data(config)
    
    # Generate STFT spectrograms
    X_specs = generate_stft_spectrograms(X_segments, config['real_test']['sr'])
    
    # Load model
    models_dir = config['real_test']['models_dir']
    if model_type == 'sound':
        model_path = os.path.join(models_dir, f'sound_seed{seed}.keras')
        model = tf.keras.models.load_model(model_path)
        y_pred_prob = model.predict(X_specs)
    elif model_type == 'mt_masknet':
        model_path = os.path.join(models_dir, f'mt_masknet_seed{seed}.keras')
        model = tf.keras.models.load_model(model_path)
        y_pred = model.predict([X_specs, X_specs])  # Duplicate input for spec/gate
        y_pred_prob = y_pred[0]  # Voice output only
    else:
        raise ValueError("Invalid model_type: choose 'sound' or 'mt_masknet'")
    
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Output only classification report
    target_names = ['WhiteBeaked', 'WhiteSided', 'AtlanticSpotted']
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config')
    parser.add_argument('--model_type', type=str, default='sound', choices=['sound', 'mt_masknet'], help='Model type')
    parser.add_argument('--seed', type=int, default=100, help='Model seed')
    args = parser.parse_args()
    test_on_real(args.config, args.model_type, args.seed)