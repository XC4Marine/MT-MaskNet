import yaml
import os
import numpy as np
from scripts.phase2_train_mt_masknet import phase2_train_mt_masknet

def main(config_path):
    """Main script to run phase 2 experiments (MT-MaskNet) for 10 seeds.
    By default, only runs for the first seed (commented loop). 
    To run all: Uncomment the for loop and indent the body accordingly.
    
    Args:
        config_path: Path to YAML config.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    results_path = config['phase2']['results_txt']
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("Phase 2: MT-MaskNet Training (10 Seeds)\n")
        f.write("="*90 + "\n\n")
    
    voice_accs = []
    
    # To run only one seed: Use the first seed
    seed = config['seeds'][0]  # First seed only
    print(f"\nRunning MT-MaskNet - Seed {seed} (single run; uncomment loop for all)")
    voice_acc = phase2_train_mt_masknet(config, seed, results_path)
    voice_accs.append(voice_acc)
    
    # Uncomment below to run for all seeds (remove the # from for and indent)
    # for seed in config['seeds']:
    #     print(f"\nRunning MT-MaskNet - Seed {seed}")
    #     voice_acc = phase2_train_mt_masknet(config, seed, results_path)
    #     voice_accs.append(voice_acc)
    
    # Calculate and append mean ± std (for voice accuracy, as primary)
    mean_acc = np.mean(voice_accs)
    std_acc = np.std(voice_accs)
    
    with open(results_path, 'a', encoding='utf-8') as f:
        f.write("\nSummary Across Seeds (Voice Accuracy):\n")
        f.write(f"Mean ± STD: {mean_acc:.4f} ± {std_acc:.4f}\n")
    
    print("\nPhase 2 complete. Results in:", results_path)
    print("Note: To build/run for all seeds, uncomment the loop in this script.")
    print("Verify layer names in src/model.py using model.summary() on a loaded phase1 model.")

if __name__ == "__main__":
    main('configs/config.yaml')