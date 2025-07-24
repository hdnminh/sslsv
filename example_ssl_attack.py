#!/usr/bin/env python3
"""
Example script demonstrating how to run SSL-based speech verification attacks
This script shows how to use the original_simple_attack_f5tts_16khz_ssl.py
"""

import os
import subprocess
import sys

def main():
    # Example SSL model configuration
    # You need to provide a valid SSL model config file and checkpoint
    ssl_config = "models/tests/simclr/simclr_e-ecapa/ssps_kmeans_25k_uni-1/config.yml"
    ssl_checkpoint = "model_avg.pt"  # or model_latest.pt
    
    # Check if SSL config exists
    if not os.path.exists(ssl_config):
        print(f"ERROR: SSL config file not found: {ssl_config}")
        print("Please provide a valid SSL model config file.")
        print("You can find SSL model configs in the models/ directory.")
        return
    
    # Example attack parameters
    tts_model = "F5TTS"  # or "CoquiTTS", "KokoroTTS", "OpenAITTS"
    max_trials = 100  # Small number for testing
    batch_size = 10   # Small batch for testing
    
    # Build the command
    cmd = [
        "python", "original_simple_attack_f5tts_16khz_ssl.py",
        "--ssl_config", ssl_config,
        "--ssl_checkpoint", ssl_checkpoint,
        "--tts", tts_model,
        "--max_trials", str(max_trials),
        "--batch_size", str(batch_size),
        "--use_seeding",
        "--seed_value", "42",
        "--use_cache",
        "--protocol_file", "voxceleb1_test_O",  # Use SSL framework naming
        "--transcript_file", "voxceleb_transcripts_test.csv",  # Relative to SSL base_path
        "--eer_method", "interpolation",  # Use more precise threshold computation
        "--output_dir", "ssl_attack_results_example",
        "--cache_dir", "cache_ssl_example"
    ]
    
    print("Running SSL Speech Verification Attack...")
    print("Command:", " ".join(cmd))
    print()
    
    # Run the attack
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\nAttack completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nAttack failed with error code: {e.returncode}")
        print("Check the logs for more details.")
    except FileNotFoundError:
        print("Error: original_simple_attack_f5tts_16khz_ssl.py not found!")
        print("Make sure you're running this from the correct directory.")

def list_available_ssl_models():
    """List available SSL model configurations"""
    print("Looking for SSL model configurations...")
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return
    
    configs = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file == "config.yml":
                config_path = os.path.join(root, file)
                configs.append(config_path)
    
    if configs:
        print(f"Found {len(configs)} SSL model configurations:")
        for config in sorted(configs):
            print(f"  - {config}")
    else:
        print("No SSL model configurations found.")
    print()

if __name__ == "__main__":
    print("=== SSL Speech Verification Attack Example ===")
    print()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--list-models":
        list_available_ssl_models()
    else:
        print("This example demonstrates how to run attacks against SSL-based speaker verification.")
        print("The script will attack SSL models to study their linguistic sensitivity.")
        print()
        
        # List available models first
        list_available_ssl_models()
        
        # Run the example
        main()
        
        print("\nTips:")
        print("1. Use --list-models to see available SSL model configurations")
        print("2. Adjust --max_trials for longer experiments (default: all trials)")
        print("3. Use F5TTS or CoquiTTS for voice cloning attacks")
        print("4. Results will be saved in JSON format with detailed metrics")
        print("5. Compare results between different SSL models to study linguistic sensitivity") 