#!/usr/bin/env python3
"""
Find optimal thresholds for SSL Speaker Verification models
This script computes EER threshold and other decision thresholds
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Set up paths so we can import notebooks_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'notebooks'))

from notebooks.notebooks_utils import load_models, evaluate_models, create_metrics_df
from sslsv.evaluations.CosineSVEvaluation import CosineSVEvaluation, CosineSVEvaluationTaskConfig


def compute_error_rates(scores, targets):
    """
    Compute error rates for all possible thresholds
    
    Returns:
        fprs: False Positive Rates
        fnrs: False Negative Rates  
        sorted_scores: Scores sorted in ascending order (these are the thresholds)
    """
    scores = np.array(scores)
    targets = np.array(targets)

    nb_target_scores = len(scores[targets == 1])
    nb_nontarget_scores = len(scores[targets == 0])

    sorted_idx = np.argsort(scores)

    # Determine the number of positives that will be classified
    # as negatives as the score/threshold increases.
    sum_fn = np.cumsum(targets[sorted_idx])

    # Determine the number of negatives that will be classified
    # as positives as the score/threshold increases.
    sum_fp = np.cumsum(np.where(targets[sorted_idx] == 0, 1, 0))

    fnrs = np.empty(len(scores) + 1)
    fnrs[0] = 0
    fnrs[1:] = sum_fn / nb_target_scores

    fprs = np.empty(len(scores) + 1)
    fprs[0] = 1
    fprs[1:] = (nb_nontarget_scores - sum_fp) / nb_nontarget_scores

    return fprs, fnrs, scores[sorted_idx]


def find_eer_threshold(fprs, fnrs, sorted_scores):
    """Find the threshold at Equal Error Rate (EER)"""
    # Find the point where FPR is closest to FNR
    idx = np.nanargmin(np.abs(fnrs - fprs))
    eer_rate = max(fprs[idx], fnrs[idx])
    eer_threshold = sorted_scores[idx] if idx < len(sorted_scores) else sorted_scores[-1]
    
    return eer_threshold, eer_rate


def find_operating_point_thresholds(fprs, fnrs, sorted_scores):
    """Find thresholds for different operating points"""
    thresholds = {}
    
    # High security (low FAR)
    far_1_percent_idx = np.where(fprs <= 0.01)[0]
    if len(far_1_percent_idx) > 0:
        idx = far_1_percent_idx[0]
        thresholds['high_security_1%_FAR'] = {
            'threshold': sorted_scores[idx] if idx < len(sorted_scores) else sorted_scores[-1],
            'far': fprs[idx],
            'frr': fnrs[idx]
        }
    
    # Medium security (5% FAR)
    far_5_percent_idx = np.where(fprs <= 0.05)[0]
    if len(far_5_percent_idx) > 0:
        idx = far_5_percent_idx[0]
        thresholds['medium_security_5%_FAR'] = {
            'threshold': sorted_scores[idx] if idx < len(sorted_scores) else sorted_scores[-1],
            'far': fprs[idx], 
            'frr': fnrs[idx]
        }
    
    # Low security (10% FAR)
    far_10_percent_idx = np.where(fprs <= 0.10)[0]
    if len(far_10_percent_idx) > 0:
        idx = far_10_percent_idx[0]
        thresholds['low_security_10%_FAR'] = {
            'threshold': sorted_scores[idx] if idx < len(sorted_scores) else sorted_scores[-1],
            'far': fprs[idx],
            'frr': fnrs[idx]
        }
    
    return thresholds


def analyze_ssl_thresholds(config_path, checkpoint_name="model_latest.pt", plot=True, save_results=True):
    """
    Analyze thresholds for SSL speaker verification model
    
    Args:
        config_path: Path to SSL model config
        checkpoint_name: Checkpoint name to load
        plot: Whether to plot DET curve
        save_results: Whether to save results to file
    """
    
    print(f"Loading SSL model from: {config_path}")
    print(f"Using checkpoint: {checkpoint_name}")
    
    # Load model
    models = load_models([config_path], checkpoint_name=checkpoint_name)

    # Patch model configs to avoid memory issues
    # for model_entry in models.values():
    #     model_entry.config.dataset.num_workers = 0
    #     model_entry.config.dataset.pin_memory = False

    # Create evaluation config
    eval_task_config = CosineSVEvaluationTaskConfig(
        __type__="sv_cosine",
        frame_length=None,
        num_frames=1,
        trials=['voxceleb1_test_O'],  # You can change this
        metrics=['eer', 'mindcf']
    )

    # Run evaluation
    print("Running speaker verification evaluation...")
    evaluate_models(models, CosineSVEvaluation, eval_task_config)

    # Get the model and evaluation results
    model_name = list(models.keys())[0]
    model_entry = models[model_name]
    
    # Access the scores and targets from the evaluation
    # We need to access them from the evaluation object
    # For now, let's create a simple evaluation instance to get scores
    eval_instance = CosineSVEvaluation(
        model=model_entry.model,
        config=model_entry.config,
        task_config=eval_task_config,
        device=model_entry.device,
        verbose=True,
        validation=False
    )
    
    # Run evaluation to get scores
    metrics = eval_instance.evaluate()
    scores = eval_instance.scores
    targets = eval_instance.targets
    
    print(f"\nEvaluation completed!")
    print(f"Total trials: {len(scores)}")
    print(f"Target trials (same speaker): {sum(targets)}")
    print(f"Non-target trials (different speakers): {len(targets) - sum(targets)}")
    
    # Compute error rates for all thresholds
    fprs, fnrs, sorted_scores = compute_error_rates(scores, targets)
    
    # Find EER threshold
    eer_threshold, eer_rate = find_eer_threshold(fprs, fnrs, sorted_scores)
    
    # Find other operating point thresholds
    operating_thresholds = find_operating_point_thresholds(fprs, fnrs, sorted_scores)
    
    # Print results
    print(f"\n" + "="*60)
    print("SPEAKER VERIFICATION THRESHOLDS")
    print("="*60)
    
    print(f"\n🎯 EER (Equal Error Rate) Threshold:")
    print(f"   Threshold: {eer_threshold:.4f}")
    print(f"   EER Rate: {eer_rate:.4f} ({eer_rate*100:.2f}%)")
    print(f"   📝 This is the most commonly used threshold")
    
    print(f"\n🔒 Security-based Thresholds:")
    for name, info in operating_thresholds.items():
        security_level = name.split('_')[0]
        far_level = name.split('_')[1]
        print(f"   {security_level.title()} Security ({far_level} FAR):")
        print(f"     Threshold: {info['threshold']:.4f}")
        print(f"     False Accept Rate: {info['far']:.4f} ({info['far']*100:.2f}%)")
        print(f"     False Reject Rate: {info['frr']:.4f} ({info['frr']*100:.2f}%)")
    
    # Show score distribution
    same_speaker_scores = [scores[i] for i in range(len(scores)) if targets[i] == 1]
    diff_speaker_scores = [scores[i] for i in range(len(scores)) if targets[i] == 0]
    
    print(f"\n📊 Score Distribution:")
    print(f"   Same Speaker Scores:")
    print(f"     Min: {min(same_speaker_scores):.4f}, Max: {max(same_speaker_scores):.4f}")
    print(f"     Mean: {np.mean(same_speaker_scores):.4f}, Std: {np.std(same_speaker_scores):.4f}")
    print(f"   Different Speaker Scores:")
    print(f"     Min: {min(diff_speaker_scores):.4f}, Max: {max(diff_speaker_scores):.4f}")
    print(f"     Mean: {np.mean(diff_speaker_scores):.4f}, Std: {np.std(diff_speaker_scores):.4f}")
    
    # Plot DET curve and score distributions
    if plot:
        create_plots(fprs, fnrs, same_speaker_scores, diff_speaker_scores, 
                    eer_threshold, eer_rate, operating_thresholds)
    
    # Save results
    if save_results:
        results = {
            'config_path': config_path,
            'checkpoint_name': checkpoint_name,
            'eer_threshold': float(eer_threshold),
            'eer_rate': float(eer_rate),
            'operating_thresholds': {k: {k2: float(v2) for k2, v2 in v.items()} 
                                   for k, v in operating_thresholds.items()},
            'score_statistics': {
                'same_speaker': {
                    'min': float(min(same_speaker_scores)),
                    'max': float(max(same_speaker_scores)),
                    'mean': float(np.mean(same_speaker_scores)),
                    'std': float(np.std(same_speaker_scores))
                },
                'different_speaker': {
                    'min': float(min(diff_speaker_scores)),
                    'max': float(max(diff_speaker_scores)),
                    'mean': float(np.mean(diff_speaker_scores)),
                    'std': float(np.std(diff_speaker_scores))
                }
            },
            'metrics': {k: float(v) for k, v in metrics.items()}
        }
        
        import json
        output_file = f"ssl_thresholds_{os.path.basename(config_path).replace('.yml', '')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
    
    return eer_threshold, eer_rate, operating_thresholds, metrics


def create_plots(fprs, fnrs, same_speaker_scores, diff_speaker_scores, 
                eer_threshold, eer_rate, operating_thresholds):
    """Create visualization plots"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. DET Curve
    ax1.loglog(fprs * 100, fnrs * 100, 'b-', linewidth=2)
    ax1.loglog([eer_rate * 100], [eer_rate * 100], 'ro', markersize=8, label=f'EER = {eer_rate*100:.2f}%')
    ax1.set_xlabel('False Positive Rate (%)')
    ax1.set_ylabel('False Negative Rate (%)')
    ax1.set_title('Detection Error Tradeoff (DET) Curve')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. ROC Curve
    ax2.plot(fprs, 1 - fnrs, 'b-', linewidth=2)
    ax2.plot([eer_rate], [1 - eer_rate], 'ro', markersize=8, label=f'EER Point')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Score Distribution
    ax3.hist(same_speaker_scores, bins=50, alpha=0.7, label='Same Speaker', color='green', density=True)
    ax3.hist(diff_speaker_scores, bins=50, alpha=0.7, label='Different Speaker', color='red', density=True)
    ax3.axvline(eer_threshold, color='blue', linestyle='--', label=f'EER Threshold = {eer_threshold:.3f}')
    ax3.set_xlabel('Similarity Score')
    ax3.set_ylabel('Density')
    ax3.set_title('Score Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Threshold Analysis
    thresholds_list = []
    names_list = []
    fars_list = []
    frrs_list = []
    
    # Add EER point
    thresholds_list.append(eer_threshold)
    names_list.append('EER')
    fars_list.append(eer_rate)
    frrs_list.append(eer_rate)
    
    # Add operating points
    for name, info in operating_thresholds.items():
        thresholds_list.append(info['threshold'])
        names_list.append(name.replace('_', '\n'))
        fars_list.append(info['far'])
        frrs_list.append(info['frr'])
    
    x = np.arange(len(names_list))
    width = 0.35
    
    ax4.bar(x - width/2, np.array(fars_list) * 100, width, label='False Accept Rate (%)', color='red', alpha=0.7)
    ax4.bar(x + width/2, np.array(frrs_list) * 100, width, label='False Reject Rate (%)', color='blue', alpha=0.7)
    ax4.set_xlabel('Operating Point')
    ax4.set_ylabel('Error Rate (%)')
    ax4.set_title('Error Rates at Different Thresholds')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names_list, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ssl_thresholds_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📊 Plots saved to: ssl_thresholds_analysis.png")
    plt.show()


def verify_with_threshold(score, threshold):
    """
    Make a verification decision given a score and threshold
    
    Args:
        score: Similarity score between two audio samples
        threshold: Decision threshold
        
    Returns:
        bool: True if samples are from same speaker, False otherwise
    """
    return score >= threshold


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Find optimal thresholds for SSL Speaker Verification models')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to SSL model config file')
    parser.add_argument('--checkpoint', type=str, default='model_latest.pt',
                       help='Checkpoint name (default: model_latest.pt)')
    parser.add_argument('--no_plot', action='store_true',
                       help='Skip plotting visualizations')
    parser.add_argument('--no_save', action='store_true',
                       help='Skip saving results to file')
    
    args = parser.parse_args()
    
    # Example usage
    print("🔍 Analyzing SSL Speaker Verification Thresholds...")
    
    try:
        eer_threshold, eer_rate, operating_thresholds, metrics = analyze_ssl_thresholds(
            config_path=args.config,
            checkpoint_name=args.checkpoint,
            plot=not args.no_plot,
            save_results=not args.no_save
        )
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"\n💡 Usage Examples:")
        print(f"   # For balanced security/usability:")
        print(f"   threshold = {eer_threshold:.4f}  # EER threshold")
        print(f"   ")
        print(f"   # To verify if two audio samples are from same speaker:")
        print(f"   similarity_score = your_ssl_model.compute_similarity(audio1, audio2)")
        print(f"   is_same_speaker = similarity_score >= {eer_threshold:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your SSL model config and checkpoint exist.") 