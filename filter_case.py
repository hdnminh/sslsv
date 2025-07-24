import json
import os
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any
import argparse
import random
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def write_results_to_txt(output_data: Dict[str, Any], output_txt: str) -> None:
    """Write analysis results to a text file in a readable format"""
    with open(output_txt, 'w') as f:
        f.write("THRESHOLD ANALYSIS RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        for label in sorted(output_data.keys()):
            f.write(f"Label {label}:\n")
            f.write("-" * 30 + "\n")
            
            # Overall statistics
            data = output_data[label]
            f.write(f"Total samples: {data['total_samples']}\n")
            f.write(f"Original scores - Mean: {data['original_scores_stats']['mean']:.6f}, ")
            f.write(f"Std: {data['original_scores_stats']['std']:.6f}\n")
            f.write(f"Fake scores - Mean: {data['fake_scores_stats']['mean']:.6f}, ")
            f.write(f"Std: {data['fake_scores_stats']['std']:.6f}\n\n")
            
            # Threshold analysis
            f.write("Below Threshold Analysis:\n")
            for threshold in sorted(data['threshold_analysis'].keys(), key=float):
                thresh_data = data['threshold_analysis'][threshold]
                f.write(f"\nThreshold {threshold}:\n")
                f.write(f"Number of cases: {thresh_data['num_cases']} ({thresh_data['percentage']:.1f}%)\n")
                
                # Original scores
                f.write(f"Original scores - Mean: {thresh_data['original_scores']['mean']:.6f}, ")
                f.write(f"Std: {thresh_data['original_scores']['std']:.6f}\n")
                
                # Fake scores
                f.write(f"Fake scores - Mean: {thresh_data['fake_scores']['mean']:.6f}, ")
                f.write(f"Std: {thresh_data['fake_scores']['std']:.6f}\n")
                
                # Drops
                f.write(f"Average drop: {thresh_data['drops']['mean']:.6f}\n")
                f.write(f"Max drop: {thresh_data['drops']['max']:.6f}\n")
                f.write(f"Min drop: {thresh_data['drops']['min']:.6f}\n")
                
                # Percentiles
                f.write("\nFake Score Distribution:\n")
                for p in sorted([int(k[1:]) for k in thresh_data['fake_score_percentiles'].keys()]):
                    value = thresh_data['fake_score_percentiles'][f'p{p}']
                    f.write(f"{p}th percentile: {value:.6f}\n")
                
                # Sample cases
                f.write("\nSample cases (first 5):\n")
                for case in thresh_data['cases'][:5]:
                    f.write(f"Original: {case['original']:.6f}, Fake: {case['fake']:.6f}, ")
                    f.write(f"Drop: {case['drop']:.6f}\n")
                    if case['text']:
                        f.write(f"Text: {case['text']}\n")
                f.write("\n" + "-"*30 + "\n")
            
            f.write("\n" + "="*50 + "\n\n")

def analyze_threshold_cases(
    results: List[Dict[str, Any]], 
    thresholds: List[float] = [0.67], 
    output_txt: str = 'all-filtered-cases-threshold_analysis_results.txt'
) -> Dict[str, Any]:
    """
    Analyze cases where fake scores fall below specified thresholds for each label
    """
    # Initialize containers for each label
    label_data: Dict[str, Dict[str, Any]] = {}
    output_data: Dict[str, Dict[str, Any]] = {}
    
    # Collect data for each label
    for record in results:
        try:
            label = str(record['label'])
            original_sim = float(record['original_similarity'])
            fake_sim = float(record['fake_similarity'])
            
            # Initialize label data if not exists
            if label not in label_data:
                label_data[label] = {
                    'total': 0,
                    'original_scores': [],
                    'fake_scores': [],
                    'below_threshold': {threshold: [] for threshold in thresholds}
                }
            
            label_data[label]['total'] += 1
            label_data[label]['original_scores'].append(original_sim)
            label_data[label]['fake_scores'].append(fake_sim)
            
            # Check each threshold
            for threshold in thresholds:
                if fake_sim < threshold:
                    label_data[label]['below_threshold'][threshold].append({
                        'original': original_sim,
                        'fake': fake_sim,
                        'drop': original_sim - fake_sim,
                        'text': record.get('target_transcript', ''),
                        'target_audio_rel': record.get('target_audio_rel', ''),
                        'ref_audio_rel': record.get('ref_audio_rel', ''),
                        'label': label
                    })
        except (ValueError, TypeError, KeyError):
            continue
    
    print("\nTHRESHOLD ANALYSIS")
    print("=" * 50)
    
    for label in sorted(label_data.keys()):
        print(f"\nLabel {label}:")
        print("-" * 30)
        
        total_samples = label_data[label]['total']
        original_scores = np.array(label_data[label]['original_scores'])
        fake_scores = np.array(label_data[label]['fake_scores'])
        
        # Initialize output data for this label
        output_data[label] = {
            'total_samples': total_samples,
            'original_scores_stats': {
                'mean': float(np.mean(original_scores)),
                'std': float(np.std(original_scores))
            },
            'fake_scores_stats': {
                'mean': float(np.mean(fake_scores)),
                'std': float(np.std(fake_scores))
            },
            'threshold_analysis': {}
        }
        
        print(f"Total samples: {total_samples}")
        print(f"Original scores - Mean: {np.mean(original_scores):.6f}, Std: {np.std(original_scores):.6f}")
        print(f"Fake scores - Mean: {np.mean(fake_scores):.6f}, Std: {np.std(fake_scores):.6f}")
        
        print("\nBelow Threshold Analysis:")
        for threshold in sorted(thresholds):
            cases = label_data[label]['below_threshold'][threshold]
            if cases:
                cases_array = np.array([(case['original'], case['fake'], case['drop']) for case in cases])
                
                threshold_stats = {
                    'num_cases': len(cases),
                    'percentage': float(len(cases) / total_samples * 100),
                    'original_scores': {
                        'mean': float(np.mean(cases_array[:,0])),
                        'std': float(np.std(cases_array[:,0]))
                    },
                    'fake_scores': {
                        'mean': float(np.mean(cases_array[:,1])),
                        'std': float(np.std(cases_array[:,1]))
                    },
                    'drops': {
                        'mean': float(np.mean(cases_array[:,2])),
                        'max': float(np.max(cases_array[:,2])),
                        'min': float(np.min(cases_array[:,2]))
                    },
                    'fake_score_percentiles': {},
                    'cases': cases
                }
                
                print(f"\nThreshold {threshold}:")
                print(f"Number of cases: {len(cases)} ({len(cases)/total_samples*100:.1f}%)")
                print(f"Original scores - Mean: {np.mean(cases_array[:,0]):.6f}, Std: {np.std(cases_array[:,0]):.6f}")
                print(f"Fake scores - Mean: {np.mean(cases_array[:,1]):.6f}, Std: {np.std(cases_array[:,1]):.6f}")
                print(f"Average drop: {np.mean(cases_array[:,2]):.6f}")
                print(f"Max drop: {np.max(cases_array[:,2]):.6f}")
                print(f"Min drop: {np.min(cases_array[:,2]):.6f}")
                
                print("\nFake Score Distribution:")
                percentiles = [5, 25, 50, 75, 95]
                for p in percentiles:
                    value = float(np.percentile(cases_array[:,1], p))
                    threshold_stats['fake_score_percentiles'][f'p{p}'] = value
                    print(f"{p}th percentile: {value:.6f}")
                
                output_data[label]['threshold_analysis'][str(threshold)] = threshold_stats
    
    # Write to text file
    write_results_to_txt(output_data, output_txt)
    print(f"Results written to {output_txt}")
    
    return output_data

def filter_label1_below_threshold(results: List[Dict[str, Any]], threshold: float = 0.675, 
                                 output_json: str = 'label1_below_067_cases.json') -> List[Dict[str, Any]]:
    """
    Filter all cases where label == 1 and fake_similarity < threshold.
    Write detailed cases (including audio paths) to a JSON file.
    """
    filtered_cases = []
    for record in results:
        try:
            label = int(record['label'])
            fake_sim = float(record['fake_similarity'])
            if label == 1 and fake_sim <= threshold:
                filtered_cases.append({
                    'original': float(record['original_similarity']),
                    'fake': fake_sim,
                    'drop': float(record['original_similarity']) - fake_sim,
                    'target_transcript': record.get('target_transcript', ''),
                    'target_audio_rel': record.get('target_audio_rel', ''),
                    'ref_audio_rel': record.get('ref_audio_rel', ''),
                    'label': label
                })
        except (ValueError, TypeError, KeyError):
            continue
    
    with open(output_json, 'w') as f:
        json.dump(filtered_cases, f, indent=2)
    print(f"Filtered {len(filtered_cases)} cases written to {output_json}")
    return filtered_cases


def filter_label0_below_threshold(results: List[Dict[str, Any]], threshold: float = 0.675, 
                                 output_json: str = 'label0_below_067_cases.json') -> List[Dict[str, Any]]:
    """
    Filter all cases where label == 1 and fake_similarity < threshold.
    Write detailed cases (including audio paths) to a JSON file.
    """
    filtered_cases = []
    for record in results:
        try:
            label = int(record['label'])
            fake_sim = float(record['fake_similarity'])
            if label == 1 and fake_sim <= threshold:
                filtered_cases.append({
                    'original': float(record['original_similarity']),
                    'fake': fake_sim,
                    'drop': float(record['original_similarity']) - fake_sim,
                    'target_transcript': record.get('target_transcript', ''),
                    'target_audio_rel': record.get('target_audio_rel', ''),
                    'ref_audio_rel': record.get('ref_audio_rel', ''),
                    'label': label
                })
        except (ValueError, TypeError, KeyError):
            continue
    
    with open(output_json, 'w') as f:
        json.dump(filtered_cases, f, indent=2)
    print(f"Filtered {len(filtered_cases)} cases written to {output_json}")
    return filtered_cases

def random_sample_below_threshold(results: List[Dict[str, Any]], threshold: float = 0.675, 
                                 sample_size: int = 350, output_json: str = 'random_350_below_threshold.json',
                                 seed: int = 42) -> List[Dict[str, Any]]:
    """
    Randomly sample cases where fake_similarity < threshold without replacement.
    Write detailed cases (including audio paths) to a JSON file.
    
    Args:
        results: List of result dictionaries
        threshold: Threshold value for fake_similarity
        sample_size: Number of samples to randomly select (default: 350)
        output_json: Output JSON filename
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        List of randomly sampled cases
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # First, collect all cases below threshold
    below_threshold_cases = []
    for record in results:
        try:
            fake_sim = float(record['fake_similarity'])
            if fake_sim < threshold:
                below_threshold_cases.append({
                    'original': float(record['original_similarity']),
                    'fake': fake_sim,
                    'drop': float(record['original_similarity']) - fake_sim,
                    'target_transcript': record.get('target_transcript', ''),
                    'target_audio_rel': record.get('target_audio_rel', ''),
                    'ref_audio_rel': record.get('ref_audio_rel', ''),
                    'label': int(record['label']),
                })
        except (ValueError, TypeError, KeyError):
            continue
    
    print(f"Found {len(below_threshold_cases)} total cases below threshold {threshold}")
    
    # Check if we have enough samples
    if len(below_threshold_cases) < sample_size:
        print(f"Warning: Only {len(below_threshold_cases)} cases available below threshold, "
              f"but {sample_size} requested. Using all available cases.")
        sampled_cases = below_threshold_cases
    else:
        # Randomly sample without replacement
        sampled_cases = random.sample(below_threshold_cases, sample_size)
    
    # Sort by fake similarity for easier analysis
    sampled_cases.sort(key=lambda x: x['fake'])
    
    # Save to JSON file
    with open(output_json, 'w') as f:
        json.dump(sampled_cases, f, indent=2)
    
    print(f"Randomly sampled {len(sampled_cases)} cases written to {output_json}")
    print(f"Fake similarity range: {sampled_cases[0]['fake']:.4f} to {sampled_cases[-1]['fake']:.4f}")
    
    # Print some statistics
    labels = [case['label'] for case in sampled_cases]
    label_0_count = labels.count(0)
    label_1_count = labels.count(1)
    print(f"Label distribution: {label_0_count} label 0, {label_1_count} label 1")
    
    return sampled_cases


def random_sample_label0_below_threshold(results: List[Dict[str, Any]], threshold: float = 0.675, 
                                        sample_size: int = 350, output_json: str = 'random_350_label0_below_threshold.json',
                                        seed: int = 42) -> List[Dict[str, Any]]:
    """
    Randomly sample cases where label == 0 and fake_similarity < threshold without replacement.
    Write detailed cases (including audio paths) to a JSON file.
    
    Args:
        results: List of result dictionaries
        threshold: Threshold value for fake_similarity
        sample_size: Number of samples to randomly select (default: 350)
        output_json: Output JSON filename
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        List of randomly sampled label 0 cases
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # First, collect all label 0 cases below threshold
    label0_below_threshold_cases = []
    for record in results:
        try:
            label = int(record['label'])
            fake_sim = float(record['fake_similarity'])
            if label == 0 and fake_sim < threshold:
                label0_below_threshold_cases.append({
                    'original': float(record['original_similarity']),
                    'fake': fake_sim,
                    'drop': float(record['original_similarity']) - fake_sim,
                    'target_transcript': record.get('target_transcript', ''),
                    'target_audio_rel': record.get('target_audio_rel', ''),
                    'ref_audio_rel': record.get('ref_audio_rel', ''),
                    'label': label,
                })
        except (ValueError, TypeError, KeyError):
            continue
    
    print(f"Found {len(label0_below_threshold_cases)} label 0 cases below threshold {threshold}")
    
    # Check if we have enough samples
    if len(label0_below_threshold_cases) < sample_size:
        print(f"Warning: Only {len(label0_below_threshold_cases)} label 0 cases available below threshold, "
              f"but {sample_size} requested. Using all available cases.")
        sampled_cases = label0_below_threshold_cases
    else:
        # Randomly sample without replacement
        sampled_cases = random.sample(label0_below_threshold_cases, sample_size)
    
    # Sort by fake similarity for easier analysis
    sampled_cases.sort(key=lambda x: x['fake'])
    
    # Save to JSON file
    with open(output_json, 'w') as f:
        json.dump(sampled_cases, f, indent=2)
    
    print(f"Randomly sampled {len(sampled_cases)} label 0 cases written to {output_json}")
    print(f"Fake similarity range: {sampled_cases[0]['fake']:.4f} to {sampled_cases[-1]['fake']:.4f}")
    
    return sampled_cases


def random_sample_label1_below_threshold(results: List[Dict[str, Any]], threshold: float = 0.675, 
                                        sample_size: int = 350, output_json: str = 'random_350_label1_below_threshold.json',
                                        seed: int = 42) -> List[Dict[str, Any]]:
    """
    Randomly sample cases where label == 1 and fake_similarity < threshold without replacement.
    Write detailed cases (including audio paths) to a JSON file.
    
    Args:
        results: List of result dictionaries
        threshold: Threshold value for fake_similarity
        sample_size: Number of samples to randomly select (default: 350)
        output_json: Output JSON filename
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        List of randomly sampled label 1 cases
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # First, collect all label 1 cases below threshold
    label1_below_threshold_cases = []
    for record in results:
        try:
            label = int(record['label'])
            fake_sim = float(record['fake_similarity'])
            if label == 1 and fake_sim < threshold:
                label1_below_threshold_cases.append({
                    'original': float(record['original_similarity']),
                    'fake': fake_sim,
                    'drop': float(record['original_similarity']) - fake_sim,
                    'target_transcript': record.get('target_transcript', ''),
                    'target_audio_rel': record.get('target_audio_rel', ''),
                    'ref_audio_rel': record.get('ref_audio_rel', ''),
                    'label': label,
                })
        except (ValueError, TypeError, KeyError):
            continue
    
    print(f"Found {len(label1_below_threshold_cases)} label 1 cases below threshold {threshold}")
    
    # Check if we have enough samples
    if len(label1_below_threshold_cases) < sample_size:
        print(f"Warning: Only {len(label1_below_threshold_cases)} label 1 cases available below threshold, "
              f"but {sample_size} requested. Using all available cases.")
        sampled_cases = label1_below_threshold_cases
    else:
        # Randomly sample without replacement
        sampled_cases = random.sample(label1_below_threshold_cases, sample_size)
    
    # Sort by fake similarity for easier analysis
    sampled_cases.sort(key=lambda x: x['fake'])
    
    # Save to JSON file
    with open(output_json, 'w') as f:
        json.dump(sampled_cases, f, indent=2)
    
    print(f"Randomly sampled {len(sampled_cases)} label 1 cases written to {output_json}")
    print(f"Fake similarity range: {sampled_cases[0]['fake']:.4f} to {sampled_cases[-1]['fake']:.4f}")
    
    return sampled_cases

def analyze_filtered_examples(filtered_cases: List[Dict[str, Any]], output_txt: str = 'filtered_examples_analysis.txt') -> Dict[str, Any]:
    """
    Analyze filtered examples including word count and other text metrics
    
    Args:
        filtered_cases: List of filtered case dictionaries
        output_txt: Output text file for analysis results
    
    Returns:
        Dictionary containing analysis results
    """
    if not filtered_cases:
        print("No filtered cases to analyze.")
        return {}
    
    analysis_results = {
        'total_cases': len(filtered_cases),
        'label_distribution': {},
        'transcript_analysis': {},
        'audio_path_analysis': {}
    }
    
    # Initialize counters
    label_counts = defaultdict(int)
    word_counts = []
    target_audio_paths = set()
    ref_audio_paths = set()
    
    # Process each case
    for case in filtered_cases:
        # Label distribution
        label = case.get('label', 'unknown')
        label_counts[label] += 1
        
        # Transcript analysis
        transcript = case.get('target_transcript', '')
        if transcript:
            words = transcript.split()
            word_counts.append(len(words))
        
        # Audio path analysis
        target_audio = case.get('target_audio_rel', '')
        ref_audio = case.get('ref_audio_rel', '')
        
        if target_audio:
            target_audio_paths.add(target_audio)
        if ref_audio:
            ref_audio_paths.add(ref_audio)
    
    # Calculate statistics
    analysis_results['label_distribution'] = dict(label_counts)
    
    # Transcript analysis
    if word_counts:
        analysis_results['transcript_analysis'] = {
            'word_count_stats': {
                'mean': float(np.mean(word_counts)),
                'std': float(np.std(word_counts)),
                'min': float(np.min(word_counts)),
                'max': float(np.max(word_counts)),
                'median': float(np.median(word_counts))
            },
            'word_counts': word_counts  # Store for plotting
        }
    
    # Audio path analysis
    analysis_results['audio_path_analysis'] = {
        'unique_target_audio_files': len(target_audio_paths),
        'unique_ref_audio_files': len(ref_audio_paths)
    }
    
    # Print and save results
    print_analysis_results(analysis_results, output_txt)
    
    # Generate word count distribution chart
    if word_counts:
        output_dir = os.path.dirname(output_txt)
        base_name = os.path.splitext(os.path.basename(output_txt))[0]
        chart_path = os.path.join(output_dir, f'{base_name}_word_count_distribution.png')
        plot_word_count_distribution(word_counts, chart_path)
    
    return analysis_results

def plot_word_count_distribution(word_counts: List[int], output_path: str) -> None:
    """
    Create a word count distribution chart
    
    Args:
        word_counts: List of word counts
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Histogram
    ax1.hist(word_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Word Count')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Word Count Distribution - Histogram')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_words = np.mean(word_counts)
    std_words = np.std(word_counts)
    median_words = np.median(word_counts)
    stats_text = f'Mean: {mean_words:.1f}\nStd: {std_words:.1f}\nMedian: {median_words:.1f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Box plot
    ax2.boxplot(word_counts, vert=True, patch_artist=True, 
                boxprops=dict(facecolor='lightgreen', alpha=0.7))
    ax2.set_ylabel('Word Count')
    ax2.set_title('Word Count Distribution - Box Plot')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics to box plot
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Word count distribution chart saved to: {output_path}")


def print_analysis_results(analysis_results: Dict[str, Any], output_txt: str) -> None:
    """
    Print and save analysis results to a text file
    """
    with open(output_txt, 'w') as f:
        f.write("FILTERED EXAMPLES ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        f.write(f"Total filtered cases: {analysis_results['total_cases']}\n\n")
        
        # Label distribution
        f.write("LABEL DISTRIBUTION\n")
        f.write("-" * 30 + "\n")
        for label, count in sorted(analysis_results['label_distribution'].items()):
            percentage = (count / analysis_results['total_cases']) * 100
            f.write(f"Label {label}: {count} cases ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Transcript analysis
        if analysis_results['transcript_analysis']:
            f.write("TRANSCRIPT ANALYSIS\n")
            f.write("-" * 30 + "\n")
            
            # Word count statistics
            word_stats = analysis_results['transcript_analysis']['word_count_stats']
            f.write("Word Count:\n")
            f.write(f"  Mean: {word_stats['mean']:.2f}\n")
            f.write(f"  Std: {word_stats['std']:.2f}\n")
            f.write(f"  Min: {word_stats['min']:.0f}\n")
            f.write(f"  Max: {word_stats['max']:.0f}\n")
            f.write(f"  Median: {word_stats['median']:.2f}\n")
            
            f.write("\n")

    
    # Also print to console
    print(f"\nFILTERED EXAMPLES ANALYSIS")
    print(f"Total filtered cases: {analysis_results['total_cases']}")
    print(f"Label distribution: {analysis_results['label_distribution']}")
    
    if analysis_results['transcript_analysis']:
        word_stats = analysis_results['transcript_analysis']['word_count_stats']
        print(f"Word count - Mean: {word_stats['mean']:.2f}, Std: {word_stats['std']:.2f}")
    
    print(f"Detailed analysis saved to {output_txt}")


def analyze_filtered_examples_by_label(filtered_cases: List[Dict[str, Any]], output_txt: str = 'filtered_examples_by_label_analysis.txt') -> Dict[str, Any]:
    """
    Analyze filtered examples separately for each label
    
    Args:
        filtered_cases: List of filtered case dictionaries
        output_txt: Output text file for analysis results
    
    Returns:
        Dictionary containing analysis results by label
    """
    if not filtered_cases:
        print("No filtered cases to analyze.")
        return {}
    
    # Separate cases by label
    cases_by_label = defaultdict(list)
    for case in filtered_cases:
        label = case.get('label', 'unknown')
        cases_by_label[label].append(case)
    
    analysis_results = {
        'total_cases': len(filtered_cases),
        'cases_by_label': dict(cases_by_label),
        'label_analysis': {}
    }
    
    # Analyze each label separately
    for label, cases in cases_by_label.items():
        print(f"\nAnalyzing label {label} cases ({len(cases)} cases)...")
        
        # Analyze this label's cases
        label_analysis = analyze_filtered_examples(
            cases, 
            output_txt=f'filtered_examples_label_{label}_analysis.txt'
        )
        
        analysis_results['label_analysis'][str(label)] = label_analysis
    
    # Write combined analysis
    with open(output_txt, 'w') as f:
        f.write("FILTERED EXAMPLES ANALYSIS BY LABEL\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total filtered cases: {analysis_results['total_cases']}\n\n")
        
        for label, cases in sorted(cases_by_label.items()):
            f.write(f"LABEL {label} ANALYSIS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Number of cases: {len(cases)}\n")
            f.write(f"Percentage of total: {(len(cases) / analysis_results['total_cases']) * 100:.1f}%\n\n")
            
            # Get analysis for this label
            label_analysis = analysis_results['label_analysis'].get(str(label), {})
            
            if label_analysis.get('transcript_analysis'):
                length_stats = label_analysis['transcript_analysis']['word_count_stats']
                f.write("Word Count Statistics:\n")
                f.write(f"  Mean: {length_stats['mean']:.2f}, Std: {length_stats['std']:.2f}\n\n")
            
            f.write("-" * 30 + "\n\n")
    
    print(f"Combined analysis by label saved to {output_txt}")
    return analysis_results

def main() -> None:
    """Main function to run the analysis"""
    parser = argparse.ArgumentParser(description='Filter and analyze cases')
    parser.add_argument('--output_dir', type=str, default='sv_ssl_attack_results_05072025_16khz/filter_results',
                       help='JSON file to analyze')
    parser.add_argument('--threshold_label_1', type=float, default=0.6,
                       help='Threshold for filtering cases')
    parser.add_argument("--threshold_label_0", type=float, default=0.516,
                       help='Threshold for filtering label 0 cases')
    parser.add_argument("--sample_size", type=int, default=350,
                       help='Sample size for random sampling')
    parser.add_argument("--analyze_filtered", action='store_true', default=True,
                       help='Analyze filtered examples including transcript length, word count, etc.')
    result_file_name = 'sv_ssl_attack_results_05072025_16khz/voxceleb_ssl_attack_results-ssl-model_avg-F5TTS-all-seeded-cached.json'

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    all_filtered_cases_json_file_label_1 = os.path.join(args.output_dir, f'all-filtered-cases_label_1_{args.threshold_label_1}.json')
    all_filtered_cases_json_file_label_0 = os.path.join(args.output_dir, f'all-filtered-cases_label_0_{args.threshold_label_0}.json')
    analysis_detail_all_data_txt_label_1 = os.path.join(args.output_dir, f'analysis_detail_filtered_cases_label_1_{args.threshold_label_1}.txt')
    analysis_detail_all_data_txt_label_0 = os.path.join(args.output_dir, f'analysis_detail_filtered_cases_label_0_{args.threshold_label_0}.txt')
    
    try:
        with open(result_file_name, 'r') as f:
            data = json.load(f)
        results = data.get('results', [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON file: {e}")
        return
    
    # Run filtering
    print("Running filtering operations...")
    filtered_cases_label_1 = filter_label1_below_threshold(results, threshold=args.threshold_label_1, output_json=all_filtered_cases_json_file_label_1)
    filtered_cases_label_0 = random_sample_label0_below_threshold(results, threshold=args.threshold_label_0, output_json=all_filtered_cases_json_file_label_0, sample_size=args.sample_size)

    # Run threshold analysis
    print("\nRunning threshold analysis...")
    analyze_threshold_cases(results, thresholds=[args.threshold_label_1], output_txt=analysis_detail_all_data_txt_label_1)
    analyze_threshold_cases(results, thresholds=[args.threshold_label_0], output_txt=analysis_detail_all_data_txt_label_0)

    # Analyze filtered examples if requested
    if args.analyze_filtered:
        print("\nAnalyzing filtered examples...")
        
        # Analyze label 1 cases separately
        if filtered_cases_label_1:
            print(f"Analyzing {len(filtered_cases_label_1)} label 1 filtered cases...")
            label_1_analysis_txt = os.path.join(args.output_dir, f'filtered_examples_label_1_analysis_{args.threshold_label_1}.txt')
            analyze_filtered_examples(filtered_cases_label_1, output_txt=label_1_analysis_txt)
        else:
            print("No label 1 cases to analyze.")
        
        # Analyze label 0 cases separately
        if filtered_cases_label_0:
            print(f"Analyzing {len(filtered_cases_label_0)} label 0 filtered cases...")
            label_0_analysis_txt = os.path.join(args.output_dir, f'filtered_examples_label_0_analysis_{args.threshold_label_0}.txt')
            analyze_filtered_examples(filtered_cases_label_0, output_txt=label_0_analysis_txt)
        else:
            print("No label 0 cases to analyze.")
    
    print(f"\nAll results saved to: {args.output_dir}")
    print("Files generated:")
    print(f"  - Filtered cases (label 1): {all_filtered_cases_json_file_label_1}")
    print(f"  - Filtered cases (label 0): {all_filtered_cases_json_file_label_0}")
    print(f"  - Threshold analysis (label 1): {analysis_detail_all_data_txt_label_1}")
    print(f"  - Threshold analysis (label 0): {analysis_detail_all_data_txt_label_0}")
    
    if args.analyze_filtered:
        if filtered_cases_label_1:
            print(f"  - Label 1 filtered examples analysis: {os.path.join(args.output_dir, f'filtered_examples_label_1_analysis_{args.threshold_label_1}.txt')}")
            print(f"  - Label 1 word count distribution chart: {os.path.join(args.output_dir, f'filtered_examples_label_1_word_count_distribution_{args.threshold_label_1}.png')}")
        if filtered_cases_label_0:
            print(f"  - Label 0 filtered examples analysis: {os.path.join(args.output_dir, f'filtered_examples_label_0_analysis_{args.threshold_label_0}.txt')}")
            print(f"  - Label 0 word count distribution chart: {os.path.join(args.output_dir, f'filtered_examples_label_0_word_count_distribution_{args.threshold_label_0}.png')}")

if __name__ == "__main__":
    main() 