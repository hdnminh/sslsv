#!/usr/bin/env python3
"""
Example script demonstrating how to find the optimal threshold for SSL models
"""

import os

def main():
    # Example SSL model configuration
    ssl_config = "models/tests/simclr/simclr_e-ecapa/ssps_kmeans_25k_uni-1/config.yml"
    ssl_checkpoint = "model_avg.pt"
    
    # Check if SSL config exists
    if not os.path.exists(ssl_config):
        print(f"ERROR: SSL config file not found: {ssl_config}")
        print("Please provide a valid SSL model config file.")
        return
    
    print("🔍 Finding optimal threshold for SSL model...")
    print(f"Config: {ssl_config}")
    print(f"Checkpoint: {ssl_checkpoint}")
    print()
    
    # Find the threshold
    import subprocess
    cmd = [
        "python", "find_ssl_threshold.py",
        "--config", ssl_config,
        "--checkpoint", ssl_checkpoint
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ Threshold analysis completed!")
        
        print("\n💡 How to use the threshold:")
        print("1. The script computed the EER (Equal Error Rate) threshold")
        print("2. This threshold gives the best balance between false accepts and false rejects")
        print("3. You can use this threshold in your applications:")
        print("   ```python")
        print("   # Load your SSL model")
        print("   similarity_score = ssl_model.compute_similarity(audio1, audio2)")
        print("   is_same_speaker = similarity_score >= eer_threshold")
        print("   ```")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Threshold analysis failed: {e}")
    except FileNotFoundError:
        print("❌ find_ssl_threshold.py not found!")

def show_threshold_explanation():
    """Explain what thresholds mean in speaker verification"""
    
    print("="*60)
    print("UNDERSTANDING SPEAKER VERIFICATION THRESHOLDS")
    print("="*60)
    
    print("\n🎯 What is a threshold?")
    print("   • SSL models output similarity scores between 0.0 and 1.0")
    print("   • Threshold determines the decision boundary:")
    print("     - Score >= Threshold → Same speaker")
    print("     - Score < Threshold → Different speakers")
    
    print("\n📊 Types of thresholds:")
    print("   • EER Threshold: Where False Accept Rate = False Reject Rate")
    print("     → Most commonly used, balanced security/usability")
    print("   • High Security: Low false accept rate (strict)")
    print("     → For high-security applications (banking, etc.)")
    print("   • Low Security: Low false reject rate (permissive)")
    print("     → For user-friendly applications")
    
    print("\n⚖️ Security vs Usability Trade-off:")
    print("   High Threshold (0.8+):")
    print("     ✅ High security (few false accepts)")
    print("     ❌ Poor usability (many false rejects)")
    print("   ")
    print("   Low Threshold (0.4-):")
    print("     ✅ Good usability (few false rejects)")  
    print("     ❌ Low security (many false accepts)")
    print("   ")
    print("   EER Threshold (~0.6-0.7):")
    print("     ⚖️ Balanced security and usability")
    
    print("\n🔢 Example thresholds in practice:")
    print("   • Banking/Security: 0.85+ (very strict)")
    print("   • Voice assistants: 0.50-0.65 (user-friendly)")
    print("   • General applications: 0.68 (common default)")
    print("   • Research/evaluation: EER threshold (optimal)")

if __name__ == "__main__":
    show_threshold_explanation()
    print("\n" + "="*60)
    print("FINDING YOUR SSL MODEL'S OPTIMAL THRESHOLD")
    print("="*60)
    main()
    
    print("\n📝 Next steps:")
    print("1. Run the SSL attack script - it will automatically use the EER threshold")
    print("2. The SSL attack script computes the EER threshold for your specific model")
    print("3. Results will show which threshold was used for attack decisions") 