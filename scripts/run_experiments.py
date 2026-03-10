import yaml
import os
import numpy as np
from scripts.train_sound import train_sound
from scripts.train_noise import train_noise

def main(config_path):
    """Main script to run phase 1 experiments for 10 seeds.
    
    Args:
        config_path: Path to YAML config.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    os.makedirs(os.path.dirname(config['paths']['results_dir']), exist_ok=True)
    results_path = config['paths']['results_dir']
    
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("Phase 1: Pre-training Sound and Noise Classifiers (10 Seeds)\n")
        f.write("="*80 + "\n\n")
    
    sound_accs = []
    noise_accs = []
    
    for seed in config['seeds']:
        # Train sound
        print(f"\nRunning Sound Classifier - Seed {seed}")
        sound_acc = train_sound(config, seed, results_path)
        sound_accs.append(sound_acc)
        
        # Train noise
        print(f"\nRunning Noise Classifier - Seed {seed}")
        noise_acc = train_noise(config, seed, results_path)
        noise_accs.append(noise_acc)
    
    # Calculate and append mean ± std
    sound_mean = np.mean(sound_accs)
    sound_std = np.std(sound_accs)
    noise_mean = np.mean(noise_accs)
    noise_std = np.std(noise_accs)
    
    with open(results_path, 'a', encoding='utf-8') as f:
        f.write("\nSummary Across 10 Seeds:\n")
        f.write(f"Sound Test Accuracy: {sound_mean:.4f} ± {sound_std:.4f}\n")
        f.write(f"Noise Test Accuracy: {noise_mean:.4f} ± {noise_std:.4f}\n")
    
    print("\nExperiments complete. Results in:", results_path)
    

if __name__ == "__main__":
    main('configs/config.yaml')