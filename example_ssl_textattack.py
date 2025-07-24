#!/usr/bin/env python3
"""
Example script for running SSL Speech Verification Attack with TextAttack Framework

This script demonstrates how to run adversarial attacks on SSL-based speech verification systems
using text perturbations + TTS generation.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("=== SSL Speech Verification Attack with TextAttack Framework ===")
    print()
    print("This example demonstrates adversarial attacks on SSL models using:")
    print("1. SSL models for speech verification (instead of supervised models)")
    print("2. TextAttack framework for generating perturbed text")
    print("3. TTS generation of fake audio from perturbed text")
    print("4. Automatic EER threshold computation for SSL models")
    print()
    
    # Example SSL model configuration
    ssl_config = "models/tests/simclr/simclr_e-ecapa/ssps_kmeans_25k_uni-1/config.yml"
    ssl_checkpoint = "model_avg.pt"
    
    # Check if SSL model exists
    if not Path(ssl_config).exists():
        print(f"⚠️  SSL model config not found: {ssl_config}")
        print("Please update the ssl_config path to a valid SSL model configuration.")
        return
    
    # Example parameters
    tts_model = "F5TTS"  # Use F5TTS for voice cloning
    max_examples = 10    # Limit for testing
    batch_size = 5       # Small batch for testing
    eer_method = "interpolation"  # Use more precise EER computation
    
    print(f"📊 Configuration:")
    print(f"   SSL Config: {ssl_config}")
    print(f"   SSL Checkpoint: {ssl_checkpoint}")
    print(f"   TTS Model: {tts_model}")
    print(f"   EER Method: {eer_method}")
    print(f"   Max Examples: {max_examples}")
    print(f"   Batch Size: {batch_size}")
    print()
    
    # Build the command
    cmd = [
        "python", "attack_sv_vox1_O_pwws_ssl.py",
        "--ssl_config", ssl_config,
        "--ssl_checkpoint", ssl_checkpoint,
        "--eer_method", eer_method,
        "--tts", tts_model,
        "--max_examples", str(max_examples),
        "--batch_size", str(batch_size),
        "--use_seeding",
        "--seed_value", "42",
        "--use_cache",
        "--skip", "False",  # Don't skip existing results for this example
        "--target_dataset_path", "sv_simple_attack_results_05072025_16khz/all-filtered-cases_0.63.json",
        "--output_dir", "ssl_textattack_results_example",
        "--cache_dir", "./cache/ssl_textattack_example",
        "--original_fake_audios_path", "./cache/original-fake-audios-05072025_16khz",
        "--voxceleb_root", "/media/volume/AudioUnlearnData1/sslsv/data/",
        "--similarity_threshold", "0.7"
    ]
    
    print("🚀 Running SSL TextAttack command:")
    print(" ".join(cmd))
    print()
    
    try:
        # Run the attack
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        print()
        print("✅ SSL TextAttack completed successfully!")
        print(f"📁 Results saved in: ssl_textattack_results_example/")
        print()
        print("🎯 Key Features:")
        print("   • Automatic EER threshold computation for SSL models")
        print("   • TextAttack framework integration for text perturbations")
        print("   • Voice cloning with F5TTS for realistic attacks")
        print("   • SSL framework compatibility")
        print("   • Deterministic seeding for reproducibility")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ SSL TextAttack failed with error code: {e.returncode}")
        print("Check the logs for more details.")
        print()
        print("🔧 Troubleshooting Tips:")
        print("1. Ensure SSL model config and checkpoint exist")
        print("2. Check that voxceleb_root path is correct")
        print("3. Verify target_dataset_path exists")
        print("4. Check GPU memory availability")
        print("5. Review the log files in ssl_textattack_results_example/logs/")
        
    except KeyboardInterrupt:
        print("⏹️  SSL TextAttack interrupted by user")
        
    except Exception as e:
        print(f"💥 Unexpected error: {e}")

if __name__ == "__main__":
    main() 