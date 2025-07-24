import json
import numpy as np
from collections import defaultdict
import re
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, normaltest, ttest_1samp, t, sem
import argparse 
import os 

def fix_json_file(file_path):
    """
    Attempt to fix common JSON formatting issues
    """
    print("Attempting to fix JSON formatting issues...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix common issues
    content = re.sub(r',\s*}', '}', content)
    content = re.sub(r',\s*]', ']', content)
    content = re.sub(r'},f[\d.]+', '}', content)
    
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Still having JSON issues after fixes: {e}")
        return None

def extract_results_from_json(file_path):
    """
    Extract results from JSON file with error handling
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get('results', [])
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        
        fixed_data = fix_json_file(file_path)
        if fixed_data:
            return fixed_data.get('results', [])
        
        print("Attempting manual extraction of results...")
        return extract_results_manually(file_path)

def extract_results_manually(file_path):
    """
    Manually extract results from the JSON file by parsing line by line
    """
    results = []
    current_record = {}
    in_results_section = False
    brace_count = 0
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            if '"results":' in line:
                in_results_section = True
                continue
            
            if not in_results_section:
                continue
                
            if line.startswith('{'):
                current_record = {}
                brace_count = 1
                continue
            
            if line.startswith('}'):
                brace_count -= 1
                if brace_count == 0 and current_record:
                    if all(key in current_record for key in ['label', 'original_similarity', 'fake_similarity']):
                        results.append(current_record)
                    current_record = {}
                continue
            
            if ':' in line and brace_count > 0:
                try:
                    if line.endswith(','):
                        line = line[:-1]
                    
                    if '"label":' in line:
                        match = re.search(r'"label":\s*(\d+)', line)
                        if match:
                            current_record['label'] = int(match.group(1))
                    elif '"original_similarity":' in line:
                        match = re.search(r'"original_similarity":\s*([\d.]+)', line)
                        if match:
                            current_record['original_similarity'] = float(match.group(1))
                    elif '"fake_similarity":' in line:
                        match = re.search(r'"fake_similarity":\s*([\d.]+)', line)
                        if match:
                            current_record['fake_similarity'] = float(match.group(1))
                    elif '"success":' in line:
                        current_record['success'] = 'true' in line.lower()
                    elif '"transcript":' in line:
                        match = re.search(r'"transcript":\s*"([^"]*)"', line)
                        if match:
                            current_record['transcript'] = match.group(1)
                except (ValueError, AttributeError):
                    continue
    
    return results

def plot_fake_score_distributions(results, output_dir='plots'):
    """
    Plot fake score distributions for each label and calculate ranges
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect scores by label
    label_scores = {}
    for record in results:
        try:
            label = record['label']
            fake_sim = float(record['fake_similarity'])
            original_sim = float(record['original_similarity'])
            
            if label not in label_scores:
                label_scores[label] = {'fake': [], 'original': []}
            
            label_scores[label]['fake'].append(fake_sim)
            label_scores[label]['original'].append(original_sim)
            
        except (ValueError, TypeError, KeyError):
            continue
    
    # Capture console output
    import io
    from contextlib import redirect_stdout
    
    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer):
        print("\nFAKE SCORE DISTRIBUTION ANALYSIS")
        print("=" * 50)
        
        # Create plots for each label
        for label in sorted(label_scores.keys()):
            fake_scores = np.array(label_scores[label]['fake'])
            original_scores = np.array(label_scores[label]['original'])
            
            # Calculate statistics
            fake_min, fake_max = np.min(fake_scores), np.max(fake_scores)
            fake_mean, fake_std = np.mean(fake_scores), np.std(fake_scores)
            original_min, original_max = np.min(original_scores), np.max(original_scores)
            original_mean, original_std = np.mean(original_scores), np.std(original_scores)
            
            print(f"\nLabel {label} Statistics:")
            print("-" * 30)
            print(f"Sample size: {len(fake_scores)}")
            print(f"Fake scores:")
            print(f"  Range: [{fake_min:.3f}, {fake_max:.3f}]")
            print(f"  Mean ± Std: {fake_mean:.3f} ± {fake_std:.3f}")
            print(f"  Median: {np.median(fake_scores):.3f}")
            print(f"Original scores:")
            print(f"  Range: [{original_min:.3f}, {original_max:.3f}]")
            print(f"  Mean ± Std: {original_mean:.3f} ± {original_std:.3f}")
            print(f"  Median: {np.median(original_scores):.3f}")
            
            # Create histogram plot
            plt.figure(figsize=(12, 8))
            
            # Plot distributions
            plt.subplot(2, 2, 1)
            plt.hist(fake_scores, bins=50, alpha=0.7, color='red', label=f'Fake (Label {label})')
            plt.axvline(float(fake_mean), color='red', linestyle='--', label=f'Mean: {fake_mean:.3f}')
            plt.xlabel('Similarity Score')
            plt.ylabel('Frequency')
            plt.title(f'Fake Score Distribution - Label {label}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            plt.hist(original_scores, bins=50, alpha=0.7, color='blue', label=f'Original (Label {label})')
            plt.axvline(float(original_mean), color='blue', linestyle='--', label=f'Mean: {original_mean:.3f}')
            plt.xlabel('Similarity Score')
            plt.ylabel('Frequency')
            plt.title(f'Original Score Distribution - Label {label}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Overlapping distributions
            plt.subplot(2, 2, 3)
            plt.hist(original_scores, bins=50, alpha=0.6, color='blue', label=f'Original (Label {label})')
            plt.hist(fake_scores, bins=50, alpha=0.6, color='red', label=f'Fake (Label {label})')
            plt.axvline(float(original_mean), color='blue', linestyle='--', alpha=0.8)
            plt.axvline(float(fake_mean), color='red', linestyle='--', alpha=0.8)
            plt.xlabel('Similarity Score')
            plt.ylabel('Frequency')
            plt.title(f'Score Distributions Comparison - Label {label}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Box plot
            plt.subplot(2, 2, 4)
            box_data = [original_scores, fake_scores]
            box_labels = ['Original', 'Fake']
            bp = plt.boxplot(box_data)
            plt.xticks([1, 2], box_labels)
            plt.ylabel('Similarity Score')
            plt.title(f'Score Distribution Box Plot - Label {label}')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/fake_score_distribution_label_{label}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calculate percentiles
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            print(f"\nFake Score Percentiles (Label {label}):")
            for p in percentiles:
                value = np.percentile(fake_scores, p)
                print(f"  {p}th percentile: {value:.3f}")
        
        # Combined plot
        plt.figure(figsize=(15, 10))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # All fake scores
        plt.subplot(2, 3, 1)
        for i, label in enumerate(sorted(label_scores.keys())):
            fake_scores = np.array(label_scores[label]['fake'])
            plt.hist(fake_scores, bins=30, alpha=0.6, color=colors[i % len(colors)], 
                    label=f'Fake Label {label}')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title('All Fake Score Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # All original scores
        plt.subplot(2, 3, 2)
        for i, label in enumerate(sorted(label_scores.keys())):
            original_scores = np.array(label_scores[label]['original'])
            plt.hist(original_scores, bins=30, alpha=0.6, color=colors[i % len(colors)], 
                    label=f'Original Label {label}')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title('All Original Score Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Difference plot
        plt.subplot(2, 3, 3)
        for i, label in enumerate(sorted(label_scores.keys())):
            fake_scores = np.array(label_scores[label]['fake'])
            original_scores = np.array(label_scores[label]['original'])
            differences = fake_scores - original_scores
            plt.hist(differences, bins=30, alpha=0.6, color=colors[i % len(colors)], 
                    label=f'Diff Label {label}')
        plt.axvline(0, color='black', linestyle='--', alpha=0.8)
        plt.xlabel('Score Difference (Fake - Original)')
        plt.ylabel('Frequency')
        plt.title('Score Difference Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Scatter plot for each label
        for idx, label in enumerate(sorted(label_scores.keys())):
            plt.subplot(2, 3, 4 + idx)
            fake_scores = np.array(label_scores[label]['fake'])
            original_scores = np.array(label_scores[label]['original'])
            
            plt.scatter(original_scores, fake_scores, alpha=0.5, s=1)
            
            # Add diagonal line (perfect correlation)
            min_val = min(np.min(original_scores), np.min(fake_scores))
            max_val = max(np.max(original_scores), np.max(fake_scores))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            plt.xlabel('Original Score')
            plt.ylabel('Fake Score')
            plt.title(f'Original vs Fake Scores - Label {label}')
            plt.grid(True, alpha=0.3)
            
            # Calculate correlation
            correlation = np.corrcoef(original_scores, fake_scores)[0, 1]
            plt.text(0.05, 0.95, f'Corr: {correlation:.3f}', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/combined_score_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nPlots saved to '{output_dir}/' directory")
    
    # Get the captured output
    console_output = output_buffer.getvalue()
    
    # Write console output to file
    console_output_file = os.path.join(output_dir, 'fake_score_distribution_console_output.txt')
    with open(console_output_file, 'w') as f:
        f.write(console_output)
    print(f"Fake score distribution console output written to {console_output_file}")
    
    return label_scores

def analyze_label_statistics(changes, stats, label):
    """
    Perform detailed statistical analysis for a specific label
    """
    drops = stats['drops']
    increases = stats['increases']
    changes_array = np.array(changes)  # Convert to numpy array for statistical calculations
    
    print(f"\nDETAILED STATISTICS FOR LABEL {label}")
    print("=" * 50)
    
    # Success Rate Analysis
    success_count = sum(1 for x in changes if x > 0)
    success_rate = success_count / len(changes) * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
def analyze_score_differences(results, output_dir):
    """
    Analyze the differences between original and fake scores for each label
    """
    # Initialize containers for each label
    label_scores = defaultdict(lambda: {'original': [], 'fake': [], 'increases': [], 'drops': []})
    
    # Collect scores for each label
    for record in results:
        try:
            label = record['label']
            original_sim = float(record['original_similarity'])
            fake_sim = float(record['fake_similarity'])
            
            label_scores[label]['original'].append(original_sim)
            label_scores[label]['fake'].append(fake_sim)
            
            diff = fake_sim - original_sim
            if diff > 0:
                label_scores[label]['increases'].append(diff)
            elif diff < 0:
                label_scores[label]['drops'].append(abs(diff))
        except (ValueError, TypeError):
            continue
    
    # Capture console output
    import io
    from contextlib import redirect_stdout
    
    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer):
        print("\nSCORE DIFFERENCE ANALYSIS")
        print("=" * 50)
    
        for label in sorted(label_scores.keys()):
            original = np.array(label_scores[label]['original'])
            fake = np.array(label_scores[label]['fake'])
            increases = np.array(label_scores[label]['increases'])
            drops = np.array(label_scores[label]['drops'])
            
            print(f"\nLabel {label}:")
            print("-" * 30)
            print(f"Total samples: {len(original)}")
            print(f"Original scores - Mean: {np.mean(original):.3f}, Std: {np.std(original):.3f}")
            print(f"Fake scores - Mean: {np.mean(fake):.3f}, Std: {np.std(fake):.3f}")
            
            # Score Increases Analysis
            print("\nINCREASES:")
            print(f"Number of increases: {len(increases)} ({len(increases)/len(original)*100:.1f}%)")
            if len(increases) > 0:
                    print(f"Average increase: {np.mean(increases):.3f}")
                    print(f"Max increase: {np.max(increases):.3f}")
                    print(f"Min increase: {np.min(increases):.3f}")
                    print(f"Median increase: {np.median(increases):.3f}")
                    print(f"Std of increases: {np.std(increases):.3f}")
            
            # Score Drops Analysis
            print("\nDROPS:")
            print(f"Number of drops: {len(drops)} ({len(drops)/len(original)*100:.1f}%)")
            if len(drops) > 0:
                    print(f"Average drop: {np.mean(drops):.3f}")
                    print(f"Max drop: {np.max(drops):.3f}")
                    print(f"Min drop: {np.min(drops):.3f}")
                    print(f"Median drop: {np.median(drops):.3f}")
                    print(f"Std of drops: {np.std(drops):.3f}")
            
            print("\n" + "-"*30)

    # Get the captured output
    console_output = output_buffer.getvalue()
    
    # Write console output to file
    console_output_file = os.path.join(output_dir, 'score_differences_console_output.txt')
    with open(console_output_file, 'w') as f:
        f.write(console_output)
    print(f"Score differences console output written to {console_output_file}")

def write_results_to_txt(output_data, output_file):
    """Write analysis results to a text file in a readable format"""
    with open(output_file, 'w') as f:
        f.write("THRESHOLD ANALYSIS RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        for label in sorted(output_data.keys()):
            f.write(f"Label {label}:\n")
            f.write("-" * 30 + "\n")
            
            # Overall statistics
            data = output_data[label]
            f.write(f"Total samples: {data['total_samples']}\n")
            f.write(f"Original scores - Mean: {data['original_scores_stats']['mean']:.3f}, ")
            f.write(f"Std: {data['original_scores_stats']['std']:.3f}\n")
            f.write(f"Fake scores - Mean: {data['fake_scores_stats']['mean']:.3f}, ")
            f.write(f"Std: {data['fake_scores_stats']['std']:.3f}\n\n")
            
            # Threshold analysis
            f.write("Below Threshold Analysis:\n")
            for threshold in sorted(data['threshold_analysis'].keys(), key=float):
                thresh_data = data['threshold_analysis'][threshold]
                f.write(f"\nThreshold {threshold}:\n")
                f.write(f"Number of cases: {thresh_data['num_cases']} ({thresh_data['percentage']:.1f}%)\n")
                
                # Original scores
                f.write(f"Original scores - Mean: {thresh_data['original_scores']['mean']:.3f}, ")
                f.write(f"Std: {thresh_data['original_scores']['std']:.3f}\n")
                
                # Fake scores
                f.write(f"Fake scores - Mean: {thresh_data['fake_scores']['mean']:.3f}, ")
                f.write(f"Std: {thresh_data['fake_scores']['std']:.3f}\n")
                
                # Drops
                f.write(f"Average drop: {thresh_data['drops']['mean']:.3f}\n")
                f.write(f"Max drop: {thresh_data['drops']['max']:.3f}\n")
                f.write(f"Min drop: {thresh_data['drops']['min']:.3f}\n")
                
                # Percentiles with sample count
                f.write("\nFake Score Distribution:\n")
                f.write(f"Sample size for percentiles: {thresh_data['num_cases']}\n")
                
                # Add percentile range analysis to the stored data
                if 'fake_score_percentiles' in thresh_data:
                    percentiles = [5, 25, 50, 75, 95]
                    f.write("\nPercentile Range Analysis:\n")
                    f.write("-" * 30 + "\n")
                    
                    # This would need to be calculated from the actual data
                    # For now, we'll just show the percentile values
                for p in sorted([int(k[1:]) for k in thresh_data['fake_score_percentiles'].keys()]):
                    value = thresh_data['fake_score_percentiles'][f'p{p}']
                    f.write(f"{p}th percentile: {value:.3f}\n")
                
                # Outlier analysis
                if 'outlier_analysis' in thresh_data:
                    outlier_data = thresh_data['outlier_analysis']
                    f.write("\nOutlier Analysis:\n")
                    f.write(f"IQR: {outlier_data['iqr']:.3f}\n")
                    f.write(f"Lower bound (Q1 - 1.5*IQR): {outlier_data['lower_bound']:.3f}\n")
                    f.write(f"Upper bound (Q3 + 1.5*IQR): {outlier_data['upper_bound']:.3f}\n")
                    f.write(f"Number of outliers: {outlier_data['num_outliers']} ({outlier_data['outlier_percentage']:.1f}%)\n")
                    f.write(f"Low outliers: {outlier_data['num_low_outliers']} cases\n")
                    f.write(f"High outliers: {outlier_data['num_high_outliers']} cases\n")
                    f.write(f"Extreme values (beyond 3*IQR): {outlier_data['num_extreme_values']} cases\n")
                    f.write(f"Extreme lower bound: {outlier_data['extreme_lower_bound']:.3f}\n")
                    f.write(f"Extreme upper bound: {outlier_data['extreme_upper_bound']:.3f}\n")
                
                # Sample cases
                f.write("\nSample cases (first 5):\n")
                for case in thresh_data['cases'][:5]:
                    f.write(f"Speaker={case.get('speaker_id', 'unknown')}, Video={case.get('video_id', 'unknown')}, File={case.get('audio_filename', 'unknown')}\n")
                    f.write(f"Original: {case['original']:.3f}, Fake: {case['fake']:.3f}, ")
                    f.write(f"Drop: {case['drop']:.3f}\n")
                    if case.get('text'):
                        f.write(f"Text: {case['text']}\n")
                f.write("\n" + "-"*30 + "\n")
            
            f.write("\n" + "="*50 + "\n\n")

def analyze_invalid_records(invalid_records, output_dir):
    """
    Analyze invalid records to understand data quality issues
    """
    if not invalid_records:
        print("No invalid records found.")
        return
    
    print("\nINVALID RECORDS ANALYSIS")
    print("=" * 50)
    
    # Categorize by error type
    error_types = {}
    label_errors = []
    data_errors = []
    
    for record in invalid_records:
        error_type = record.get('type', 'unknown')
        if error_type not in error_types:
            error_types[error_type] = 0
        error_types[error_type] += 1
        
        if error_type == 'label_error':
            label_errors.append(record)
        elif error_type == 'data_error':
            data_errors.append(record)
    
    print(f"Total invalid records: {len(invalid_records)}")
    print("\nError Type Breakdown:")
    for error_type, count in error_types.items():
        print(f"  {error_type}: {count} records ({count/len(invalid_records)*100:.1f}%)")
    
    # Analyze ALL invalid records for label information (not just label errors)
    print(f"\nLabel Analysis for ALL Invalid Records ({len(invalid_records)} records):")
    print("-" * 50)
    
    # Try to extract label information from all invalid records
    label_attempts = []
    label_extraction_errors = 0
    
    for record in invalid_records:
        try:
            # Try different possible field names for label
            original_record = record['record']
            label = None
            
            # Try common label field names
            for field_name in ['label', 'Label', 'LABEL']:
                if field_name in original_record:
                    label = original_record[field_name]
                    break
            
            if label is not None:
                label_attempts.append(label)
            else:
                label_extraction_errors += 1
                
        except Exception as e:
            label_extraction_errors += 1
    
    if label_attempts:
        unique_labels = set(label_attempts)
        print(f"Successfully extracted labels from {len(label_attempts)} records")
        print(f"Failed to extract labels from {label_extraction_errors} records")
        print(f"Found labels: {sorted(unique_labels)}")
        
        # Count by label
        label_counts = {}
        for label in label_attempts:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        
        print("\nLabel distribution in invalid records:")
        for label in sorted(label_counts.keys()):
            print(f"  Label {label}: {label_counts[label]} records ({label_counts[label]/len(invalid_records)*100:.1f}%)")
    else:
        print("Could not extract any label information from invalid records")
    
    # Analyze label errors specifically
    if label_errors:
        print(f"\nLabel Error Analysis ({len(label_errors)} records):")
        print("-" * 30)
        
        # Try to extract label information from the record
        label_attempts_label_errors = []
        for record in label_errors:
            try:
                # Try different possible field names
                label = record['record'].get('label', None)
                if label is not None:
                    label_attempts_label_errors.append(label)
            except:
                pass
        
        if label_attempts_label_errors:
            unique_labels = set(label_attempts_label_errors)
            print(f"Attempted to extract labels from {len(label_attempts_label_errors)} label error records")
            print(f"Found labels: {sorted(unique_labels)}")
            
            # Count by label
            label_counts = {}
            for label in label_attempts_label_errors:
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
            
            print("Label distribution in label error records:")
            for label in sorted(label_counts.keys()):
                print(f"  Label {label}: {label_counts[label]} records")
    
    # Analyze data errors
    if data_errors:
        print(f"\nData Error Analysis ({len(data_errors)} records):")
        print("-" * 30)
        
        # Categorize data errors
        error_messages = {}
        for record in data_errors:
            error_msg = record.get('error', 'unknown')
            if error_msg not in error_messages:
                error_messages[error_msg] = 0
            error_messages[error_msg] += 1
        
        print("Data error types:")
        for error_msg, count in error_messages.items():
            print(f"  {error_msg}: {count} records")
    
    # Save detailed analysis to file
    analysis_file = os.path.join(output_dir, 'invalid_records_analysis.txt')
    with open(analysis_file, 'w') as f:
        f.write("INVALID RECORDS ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total invalid records: {len(invalid_records)}\n\n")
        
        f.write("Error Type Breakdown:\n")
        for error_type, count in error_types.items():
            f.write(f"  {error_type}: {count} records ({count/len(invalid_records)*100:.1f}%)\n")
        
        f.write(f"\nLabel Analysis for ALL Invalid Records ({len(invalid_records)} records):\n")
        f.write("-" * 50 + "\n")
        if label_attempts:
            f.write(f"Successfully extracted labels from {len(label_attempts)} records\n")
            f.write(f"Failed to extract labels from {label_extraction_errors} records\n")
            f.write(f"Found labels: {sorted(unique_labels)}\n\n")
            f.write("Label distribution in invalid records:\n")
            for label in sorted(label_counts.keys()):
                f.write(f"  Label {label}: {label_counts[label]} records ({label_counts[label]/len(invalid_records)*100:.1f}%)\n")
        else:
            f.write("Could not extract any label information from invalid records\n")
        
        if label_errors:
            f.write(f"\nLabel Error Analysis ({len(label_errors)} records):\n")
            f.write("-" * 30 + "\n")
            if label_attempts_label_errors:
                f.write(f"Attempted to extract labels from {len(label_attempts_label_errors)} records\n")
                f.write(f"Found labels: {sorted(unique_labels)}\n\n")
                f.write("Label distribution in label error records:\n")
                for label in sorted(label_counts.keys()):
                    f.write(f"  Label {label}: {label_counts[label]} records\n")
        
        if data_errors:
            f.write(f"\nData Error Analysis ({len(data_errors)} records):\n")
            f.write("-" * 30 + "\n")
            f.write("Data error types:\n")
            for error_msg, count in error_messages.items():
                f.write(f"  {error_msg}: {count} records\n")
    
    print(f"\nDetailed invalid records analysis saved to {analysis_file}")

def analyze_threshold_cases(results, thresholds=[0.67], output_dir = 'plots'):
    """
    Analyze cases where fake scores fall below specified thresholds for each label
    """
    # Initialize containers for each label - fix the defaultdict structure
    label_data = {}
    
    # For JSON output
    output_data = {}
    
    # Track invalid records
    invalid_records = []
    
    # Capture console output
    import io
    import sys
    from contextlib import redirect_stdout
    
    # First pass: identify all unique labels and initialize data structures
    unique_labels = set()
    for record in results:
        try:
            label = record['label']
            unique_labels.add(label)
        except (KeyError, TypeError):
            invalid_records.append({
                'record': record,
                'error': 'Missing or invalid label',
                'type': 'label_error'
            })
            continue
    
    # Initialize data structures for all labels at once
    for label in unique_labels:
        label_data[label] = {
            'total': 0,
            'original_scores': [],
            'fake_scores': [],
            'below_threshold': {}
        }
        # Initialize threshold tracking
        for threshold in thresholds:
            label_data[label]['below_threshold'][threshold] = []
    
    # Second pass: collect data for each label
    for record in results:
        try:
            label = record['label']
            original_sim = float(record['original_similarity'])
            fake_sim = float(record['fake_similarity'])
            
            label_data[label]['total'] += 1
            label_data[label]['original_scores'].append(original_sim)
            label_data[label]['fake_scores'].append(fake_sim)
            
            # Check each threshold
            for threshold in thresholds:
                if fake_sim < threshold:
                    # Extract path information for better identification
                    target_audio_rel = record.get('target_audio_rel', '')
                    path_parts = target_audio_rel.split('/')
                    speaker_id = path_parts[0] if len(path_parts) > 0 else 'unknown'
                    video_id = path_parts[1] if len(path_parts) > 1 else 'unknown'
                    audio_filename = path_parts[2] if len(path_parts) > 2 else 'unknown'
                    
                    label_data[label]['below_threshold'][threshold].append({
                        'original': original_sim,
                        'fake': fake_sim,
                        'drop': original_sim - fake_sim,
                        'text': record.get('target_transcript', ''),  # Store text for reference
                        'original_audio': record.get('ref_audio_rel', ''),
                        'fake_audio': record.get('target_audio_rel', ''),
                        'speaker_id': speaker_id,
                        'video_id': video_id,
                        'audio_filename': audio_filename,
                        'label': label
                    })
        except (ValueError, TypeError) as e:
            invalid_records.append({
                'record': record,
                'error': str(e),
                'type': 'data_error'
            })
            continue
    
    # Save invalid records to file
    invalid_records_file = os.path.join(output_dir, 'invalid_records.json')
    with open(invalid_records_file, 'w') as f:
        json.dump(invalid_records, f, indent=2)
    print(f"Invalid records saved to {invalid_records_file}")
    print(f"Total invalid records: {len(invalid_records)}")
    
    # Capture console output
    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer):
        print("\nTHRESHOLD ANALYSIS")
        print("=" * 50)
    
        for label in sorted(label_data.keys()):
            print(f"\nLabel {label}:")
            print("-" * 30)
            
            total_samples = label_data[label]['total']
            original_scores = np.array(label_data[label]['original_scores'])
            fake_scores = np.array(label_data[label]['fake_scores'])
            
            # Store basic stats
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
            print(f"Original scores - Mean: {np.mean(original_scores):.3f}, Std: {np.std(original_scores):.3f}")
            print(f"Fake scores - Mean: {np.mean(fake_scores):.3f}, Std: {np.std(fake_scores):.3f}")
            
            print("\nBelow Threshold Analysis:")
            for threshold in sorted(thresholds):
                cases = label_data[label]['below_threshold'][threshold]
                if cases:
                    cases_array = np.array([(case['original'], case['fake'], case['drop']) for case in cases])
                    
                    threshold_stats = {
                        'num_cases': len(cases),
                        'percentage': float(len(cases)/total_samples*100),
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
                        'cases': cases  # Store individual cases for detailed analysis
                    }
                    
                    print(f"\nThreshold {threshold}:")
                    print(f"Number of cases: {len(cases)} ({len(cases)/total_samples*100:.1f}%)")
                    print(f"Original scores - Mean: {np.mean(cases_array[:,0]):.3f}, Std: {np.std(cases_array[:,0]):.3f}")
                    print(f"Fake scores - Mean: {np.mean(cases_array[:,1]):.3f}, Std: {np.std(cases_array[:,1]):.3f}")
                    print(f"Average drop: {np.mean(cases_array[:,2]):.3f}")
                    print(f"Max drop: {np.max(cases_array[:,2]):.3f}")
                    print(f"Min drop: {np.min(cases_array[:,2]):.3f}")
                    
                    print("\nFake Score Distribution:")
                    percentiles = [5, 25, 50, 75, 95]
                    print(f"Sample size for percentiles: {len(cases)}")
                    
                    # Calculate actual counts for each percentile range
                    sorted_scores = np.sort(cases_array[:,1])
                    print("\nPercentile Range Analysis:")
                    print("-" * 30)
                    
                    for i, p in enumerate(percentiles):
                        value = float(np.percentile(cases_array[:,1], p))
                        threshold_stats['fake_score_percentiles'][f'p{p}'] = value
                        
                        if i == 0:
                            # Below 5th percentile
                            count_below = int(np.sum(cases_array[:,1] < value))
                            print(f"Below {p}th percentile ({value:.3f}): {count_below} samples")
                        else:
                            # Between previous percentile and current percentile
                            prev_p = percentiles[i-1]
                            prev_value = float(np.percentile(cases_array[:,1], prev_p))
                            count_in_range = int(np.sum((cases_array[:,1] >= prev_value) & (cases_array[:,1] < value)))
                            print(f"{prev_p}th-{p}th percentile ({prev_value:.3f}-{value:.3f}): {count_in_range} samples")
                        
                    # Above 95th percentile
                    p95_value = float(np.percentile(cases_array[:,1], 95))
                    count_above = int(np.sum(cases_array[:,1] >= p95_value))
                    print(f"Above 95th percentile ({p95_value:.3f}): {count_above} samples")
                    
                    # Also show the percentile values for reference
                    print(f"\nPercentile Values:")
                    for p in percentiles:
                        value = float(np.percentile(cases_array[:,1], p))
                        print(f"{p}th percentile: {value:.3f}")
                    
                    # Outlier detection for troubleshooting
                    print("\nOutlier Analysis:")
                    print("-" * 20)
                    
                    # Calculate IQR for outlier detection
                    q1 = np.percentile(cases_array[:,1], 25)
                    q3 = np.percentile(cases_array[:,1], 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    # Find outliers
                    outliers_low = cases_array[:,1] < lower_bound
                    outliers_high = cases_array[:,1] > upper_bound
                    outliers_total = outliers_low | outliers_high
                    
                    print(f"IQR: {iqr:.3f}")
                    print(f"Lower bound (Q1 - 1.5*IQR): {lower_bound:.3f}")
                    print(f"Upper bound (Q3 + 1.5*IQR): {upper_bound:.3f}")
                    print(f"Number of outliers: {np.sum(outliers_total)} ({np.sum(outliers_total)/len(cases)*100:.1f}%)")
                    
                    if np.sum(outliers_low) > 0:
                        print(f"Low outliers: {np.sum(outliers_low)} cases")
                        low_outlier_cases = [case for i, case in enumerate(cases) if outliers_low[i]]
                        print("Sample low outliers (first 3):")
                        for i, case in enumerate(low_outlier_cases[:3]):
                            print(f"  Case {i+1}: Speaker={case['speaker_id']}, Video={case['video_id']}, File={case['audio_filename']}")
                            print(f"    Original={case['original']:.3f}, Fake={case['fake']:.3f}, Drop={case['drop']:.3f}")
                            if case['text']:
                                print(f"    Text: {case['text'][:100]}...")
                    
                    if np.sum(outliers_high) > 0:
                        print(f"High outliers: {np.sum(outliers_high)} cases")
                        high_outlier_cases = [case for i, case in enumerate(cases) if outliers_high[i]]
                        print("Sample high outliers (first 3):")
                        for i, case in enumerate(high_outlier_cases[:3]):
                            print(f"  Case {i+1}: Speaker={case['speaker_id']}, Video={case['video_id']}, File={case['audio_filename']}")
                            print(f"    Original={case['original']:.3f}, Fake={case['fake']:.3f}, Drop={case['drop']:.3f}")
                            if case['text']:
                                print(f"    Text: {case['text'][:100]}...")
                    
                    # Extreme values (beyond 3*IQR)
                    extreme_lower = q1 - 3 * iqr
                    extreme_upper = q3 + 3 * iqr
                    extreme_low = cases_array[:,1] < extreme_lower
                    extreme_high = cases_array[:,1] > extreme_upper
                    extreme_total = extreme_low | extreme_high
                    
                    if np.sum(extreme_total) > 0:
                        print(f"\nExtreme values (beyond 3*IQR): {np.sum(extreme_total)} cases")
                        print(f"Extreme lower bound: {extreme_lower:.3f}")
                        print(f"Extreme upper bound: {extreme_upper:.3f}")
                        extreme_cases = [case for i, case in enumerate(cases) if extreme_total[i]]
                        print("Extreme cases (first 2):")
                        for i, case in enumerate(extreme_cases[:2]):
                            print(f"  Case {i+1}: Speaker={case['speaker_id']}, Video={case['video_id']}, File={case['audio_filename']}")
                            print(f"    Original={case['original']:.3f}, Fake={case['fake']:.3f}, Drop={case['drop']:.3f}")
                            if case['text']:
                                print(f"    Text: {case['text'][:100]}...")
                    
                    # Store outlier information in threshold_stats
                    threshold_stats['outlier_analysis'] = {
                        'iqr': float(iqr),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound),
                        'num_outliers': int(np.sum(outliers_total)),
                        'outlier_percentage': float(np.sum(outliers_total)/len(cases)*100),
                        'num_low_outliers': int(np.sum(outliers_low)),
                        'num_high_outliers': int(np.sum(outliers_high)),
                        'extreme_lower_bound': float(extreme_lower),
                        'extreme_upper_bound': float(extreme_upper),
                        'num_extreme_values': int(np.sum(extreme_total))
                    }
            
            output_data[label]['threshold_analysis'][str(threshold)] = threshold_stats
    
    # Get the captured output
    console_output = output_buffer.getvalue()
    
    # Write console output to file
    console_output_file = os.path.join(output_dir, 'threshold_analysis_console_output.txt')
    with open(console_output_file, 'w') as f:
        f.write(console_output)
    print(f"Console output written to {console_output_file}")
    
    # Write results to files
    json_file = os.path.join(output_dir, 'threshold_analysis_results.json')
    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults written to {json_file}")
    
    # Write to text file
    write_results_to_txt(output_data, os.path.join(output_dir, 'threshold_analysis_results.txt'))
    print(f"Results written to {os.path.join(output_dir, 'threshold_analysis_results.txt')}")
    
    return output_data, invalid_records

def analyze_similarity_changes(json_file_path, output_dir = 'plots'):
    """
    Analyze cosine similarity changes for each label (0, 1)
    """
    print(f"Reading file: {json_file_path}")
    
    results = extract_results_from_json(json_file_path)
    
    if not results:
        print("No results could be extracted from the file.")
        return None, None
    
    print(f"Total records extracted: {len(results)}")
    
    # Plot fake score distributions and get ranges
    label_score_data = plot_fake_score_distributions(results, output_dir)
    
    # Analyze threshold cases first and get invalid records
    threshold_results, invalid_records = analyze_threshold_cases(results, output_dir=output_dir)
    
    # Analyze score differences
    analyze_score_differences(results, output_dir)
    
    # Analyze invalid records
    analyze_invalid_records(invalid_records, output_dir)
    
    label_changes = defaultdict(list)
    label_stats = defaultdict(lambda: {'drops': [], 'increases': []})
    
    valid_records = 0
    for record in results:
        try:
            label = record['label']
            original_sim = record['original_similarity']
            fake_sim = record['fake_similarity']
            
            similarity_change = fake_sim - original_sim
            label_changes[label].append(similarity_change)
            
            if similarity_change < 0:
                label_stats[label]['drops'].append(abs(similarity_change))
            elif similarity_change > 0:
                label_stats[label]['increases'].append(similarity_change)
            
            valid_records += 1
            
        except (KeyError, TypeError, ValueError) as e:
            continue
    
    print(f"\nValid records processed: {valid_records}")
    
    # Perform detailed analysis for each label
    for label in sorted(label_changes.keys()):
        analyze_label_statistics(label_changes[label], label_stats[label], label)
    
    return label_changes, label_stats


def arg_parser():
    parser = argparse.ArgumentParser(description='Analyze similarity changes')
    parser.add_argument('--json_file', type=str, default = 'sv_ssl_attack_results_05072025_16khz/voxceleb_ssl_attack_results-ssl-model_avg-F5TTS-all-seeded-cached.json', help='Path to the JSON file')
    parser.add_argument('--output_dir', type=str, default = 'plots', help='Path to the output directory')
    return parser.parse_args()

def main():

    args = arg_parser()
    json_file = args.json_file
    output_dir = args.output_dir
    
    output_dir = os.path.join(os.path.dirname(json_file), "analysis_" + os.path.basename(json_file).split('.')[0])
    
    try:
        analyze_similarity_changes(json_file, output_dir)
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        print("Please make sure the file exists in the specified path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 