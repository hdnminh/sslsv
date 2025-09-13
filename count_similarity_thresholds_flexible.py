#!/usr/bin/env python3
"""
Comprehensive similarity analysis script that counts samples above thresholds
and generates detailed reports with export capabilities.
Can handle multiple JSON files and custom thresholds.
"""

# python count_similarity_thresholds_flexible.py real_attack_results/*.json --detailed 

import json
import os
import argparse
import glob
import csv
from datetime import datetime
from collections import defaultdict
import statistics

def analyze_json_file(json_file_path):
    """
    Analyze a single JSON file and extract all relevant information.
    
    Args:
        json_file_path (str): Path to the JSON file
    
    Returns:
        dict: Analysis results for the file
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        
        # Extract metadata
        file_info = {
            'filename': os.path.basename(json_file_path),
            'model_name': data.get('model_name', 'Unknown'),
            'tts_model': data.get('TTS_model', 'Unknown'),
            'attacker': data.get('attacker', 'Unknown'),
            'attack_strategy': data.get('attack_strategy', 'Unknown'),
            'batch_info': data.get('batch_info', {}),
            'total_samples': len(results)
        }
        
        # Extract similarity data
        original_similarities = [r.get('original_similarity', 0) for r in results]
        perturbed_similarities = [r.get('perturbed_similarity', 0) for r in results]
        improvements = [r.get('similarity_improvement', 0) for r in results]
        
        return {
            'file_info': file_info,
            'original_similarities': original_similarities,
            'perturbed_similarities': perturbed_similarities,
            'improvements': improvements,
            'success': True
        }
        
    except Exception as e:
        return {
            'file_info': {'filename': os.path.basename(json_file_path), 'error': str(e)},
            'success': False
        }

def calculate_threshold_stats(similarities, thresholds):
    """Calculate statistics for given thresholds."""
    stats = {}
    total = len(similarities)
    
    for threshold in thresholds:
        count = sum(1 for s in similarities if s > threshold)
        percentage = (count / total * 100) if total > 0 else 0
        stats[threshold] = {'count': count, 'percentage': percentage}
    
    return stats

def generate_summary_stats(all_data):
    """Generate comprehensive summary statistics."""
    total_samples = sum(data['file_info']['total_samples'] for data in all_data if data['success'])
    all_original = []
    all_perturbed = []
    all_improvements = []
    
    for data in all_data:
        if data['success']:
            all_original.extend(data['original_similarities'])
            all_perturbed.extend(data['perturbed_similarities'])
            all_improvements.extend(data['improvements'])
    
    if not all_original:
        return None
    
    summary = {
        'total_files': len([d for d in all_data if d['success']]),
        'total_samples': total_samples,
        'original_similarity': {
            'mean': statistics.mean(all_original),
            'median': statistics.median(all_original),
            'std': statistics.stdev(all_original) if len(all_original) > 1 else 0,
            'min': min(all_original),
            'max': max(all_original)
        },
        'perturbed_similarity': {
            'mean': statistics.mean(all_perturbed),
            'median': statistics.median(all_perturbed),
            'std': statistics.stdev(all_perturbed) if len(all_perturbed) > 1 else 0,
            'min': min(all_perturbed),
            'max': max(all_perturbed)
        },
        'improvement': {
            'mean': statistics.mean(all_improvements),
            'median': statistics.median(all_improvements),
            'std': statistics.stdev(all_improvements) if len(all_improvements) > 1 else 0,
            'min': min(all_improvements),
            'max': max(all_improvements)
        }
    }
    
    return summary

def print_simple_report(all_data, thresholds):
    """Print a simple threshold analysis report (original functionality)."""
    print("=" * 60)
    print("SIMILARITY THRESHOLD ANALYSIS")
    print("=" * 60)
    print(f"Thresholds: {thresholds}")
    print(f"Files to process: {len([d for d in all_data if d['success']])}")
    
    # Process each file
    total_counts = {threshold: 0 for threshold in thresholds}
    total_samples = 0
    
    for data in all_data:
        if not data['success']:
            print(f"\nWarning: Failed to process {data['file_info']['filename']}")
            continue
            
        info = data['file_info']
        print(f"\nAnalyzing: {info['filename']}")
        print(f"Total samples in file: {info['total_samples']}")
        print("-" * 60)
        
        # Count samples above each threshold
        threshold_stats = calculate_threshold_stats(data['perturbed_similarities'], thresholds)
        for threshold in thresholds:
            stats = threshold_stats[threshold]
            total_counts[threshold] += stats['count']
            print(f"Samples with perturbed_similarity > {threshold}: {stats['count']:3d} ({stats['percentage']:5.1f}%)")
        
        total_samples += info['total_samples']
    
    # Print overall summary
    if total_samples > 0:
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY:")
        print("=" * 60)
        print(f"Processed files: {len([d for d in all_data if d['success']])}")
        print(f"Total samples: {total_samples}")
        print("-" * 60)
        for threshold in thresholds:
            count = total_counts[threshold]
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            print(f"Total samples with perturbed_similarity > {threshold}: {count:3d} ({percentage:5.1f}%)")

def print_detailed_report(all_data, thresholds, summary_stats):
    """Print a comprehensive detailed report."""
    print("=" * 80)
    print("COMPREHENSIVE SIMILARITY ANALYSIS REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Thresholds analyzed: {thresholds}")
    print()
    
    # Overall statistics
    if summary_stats:
        print("OVERALL STATISTICS:")
        print("-" * 40)
        print(f"Total files processed: {summary_stats['total_files']}")
        print(f"Total samples: {summary_stats['total_samples']}")
        print()
        
        print("SIMILARITY STATISTICS:")
        print("-" * 40)
        print(f"Original Similarity  - Mean: {summary_stats['original_similarity']['mean']:.4f} ± {summary_stats['original_similarity']['std']:.4f}")
        print(f"                     - Range: [{summary_stats['original_similarity']['min']:.4f}, {summary_stats['original_similarity']['max']:.4f}]")
        print(f"Perturbed Similarity - Mean: {summary_stats['perturbed_similarity']['mean']:.4f} ± {summary_stats['perturbed_similarity']['std']:.4f}")
        print(f"                     - Range: [{summary_stats['perturbed_similarity']['min']:.4f}, {summary_stats['perturbed_similarity']['max']:.4f}]")
        print(f"Average Improvement  - Mean: {summary_stats['improvement']['mean']:.4f} ± {summary_stats['improvement']['std']:.4f}")
        print(f"                     - Range: [{summary_stats['improvement']['min']:.4f}, {summary_stats['improvement']['max']:.4f}]")
        print()
    
    # Per-file analysis
    print("PER-FILE ANALYSIS:")
    print("-" * 80)
    
    for data in all_data:
        if not data['success']:
            print(f"❌ {data['file_info']['filename']}: {data['file_info'].get('error', 'Unknown error')}")
            continue
            
        info = data['file_info']
        print(f"\n📁 {info['filename']}")
        print(f"   Model: {info.get('model_name', 'Unknown')} | TTS: {info.get('tts_model', 'Unknown')} | Attacker: {info.get('attacker', 'Unknown')}")
        print(f"   Samples: {info['total_samples']}")
        
        # Batch info if available
        batch_info = info.get('batch_info', {})
        if batch_info:
            batch_num = batch_info.get('batch_number', 'Unknown')
            total_batches = batch_info.get('total_batches', 'Unknown')
            print(f"   Batch: {batch_num}/{total_batches}")
            if 'batch_runtime_formatted' in batch_info:
                print(f"   Runtime: {batch_info['batch_runtime_formatted']}")
        
        # Threshold analysis
        threshold_stats = calculate_threshold_stats(data['perturbed_similarities'], thresholds)
        print("   Threshold Results:")
        for threshold in thresholds:
            stats = threshold_stats[threshold]
            print(f"     > {threshold}: {stats['count']:3d} samples ({stats['percentage']:5.1f}%)")
    
    # Aggregated threshold analysis
    print("\n" + "=" * 80)
    print("AGGREGATED THRESHOLD ANALYSIS:")
    print("=" * 80)
    
    all_perturbed = []
    for data in all_data:
        if data['success']:
            all_perturbed.extend(data['perturbed_similarities'])
    
    if all_perturbed:
        total_samples = len(all_perturbed)
        threshold_stats = calculate_threshold_stats(all_perturbed, thresholds)
        
        print(f"Total samples across all files: {total_samples}")
        print("-" * 40)
        for threshold in thresholds:
            stats = threshold_stats[threshold]
            print(f"Samples with perturbed_similarity > {threshold}: {stats['count']:3d} ({stats['percentage']:5.1f}%)")
    
    # Per-batch comparison
    batches = defaultdict(list)
    for data in all_data:
        if data['success'] and 'batch_info' in data['file_info']:
            batch_num = data['file_info']['batch_info'].get('batch_number')
            if batch_num:
                batches[batch_num].append(data)
    
    if len(batches) > 1:
        print("\n" + "=" * 80)
        print("PER-BATCH COMPARISON:")
        print("=" * 80)
        print(f"{'Batch':<8} {'> 0.675':<10} {'> 0.68':<10} {'> 0.7':<10}")
        print("-" * 40)
        
        for batch_num in sorted(batches.keys()):
            batch_data = batches[batch_num]
            all_similarities = []
            for data in batch_data:
                all_similarities.extend(data['perturbed_similarities'])
            
            if all_similarities:
                stats = calculate_threshold_stats(all_similarities, [0.675, 0.68, 0.7])
                print(f"Batch {batch_num:<3} {stats[0.675]['count']:2d} ({stats[0.675]['percentage']:4.1f}%) "
                      f"{stats[0.68]['count']:2d} ({stats[0.68]['percentage']:4.1f}%) "
                      f"{stats[0.7]['count']:2d} ({stats[0.7]['percentage']:4.1f}%)")

def export_to_csv(all_data, thresholds, output_file):
    """Export detailed results to CSV."""
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ['filename', 'model_name', 'tts_model', 'attacker', 'batch_number', 
                 'total_samples', 'avg_original_sim', 'avg_perturbed_sim', 'avg_improvement']
        for threshold in thresholds:
            header.extend([f'count_above_{threshold}', f'percent_above_{threshold}'])
        writer.writerow(header)
        
        # Write data
        for data in all_data:
            if not data['success']:
                continue
                
            info = data['file_info']
            threshold_stats = calculate_threshold_stats(data['perturbed_similarities'], thresholds)
            
            row = [
                info['filename'],
                info.get('model_name', ''),
                info.get('tts_model', ''),
                info.get('attacker', ''),
                info.get('batch_info', {}).get('batch_number', ''),
                info['total_samples'],
                statistics.mean(data['original_similarities']) if data['original_similarities'] else 0,
                statistics.mean(data['perturbed_similarities']) if data['perturbed_similarities'] else 0,
                statistics.mean(data['improvements']) if data['improvements'] else 0
            ]
            
            for threshold in thresholds:
                stats = threshold_stats[threshold]
                row.extend([stats['count'], stats['percentage']])
            
            writer.writerow(row)

def export_summary_report(all_data, thresholds, summary_stats, output_file):
    """Export summary report to text file."""
    with open(output_file, 'w') as f:
        f.write("SIMILARITY ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if summary_stats:
            f.write(f"Total Files: {summary_stats['total_files']}\n")
            f.write(f"Total Samples: {summary_stats['total_samples']}\n\n")
            
            f.write("THRESHOLD ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            
            all_perturbed = []
            for data in all_data:
                if data['success']:
                    all_perturbed.extend(data['perturbed_similarities'])
            
            if all_perturbed:
                threshold_stats = calculate_threshold_stats(all_perturbed, thresholds)
                for threshold in thresholds:
                    stats = threshold_stats[threshold]
                    f.write(f"Samples > {threshold}: {stats['count']} ({stats['percentage']:.1f}%)\n")
            
            f.write(f"\nAVERAGE STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Original Similarity: {summary_stats['original_similarity']['mean']:.4f} ± {summary_stats['original_similarity']['std']:.4f}\n")
            f.write(f"Perturbed Similarity: {summary_stats['perturbed_similarity']['mean']:.4f} ± {summary_stats['perturbed_similarity']['std']:.4f}\n")
            f.write(f"Average Improvement: {summary_stats['improvement']['mean']:.4f} ± {summary_stats['improvement']['std']:.4f}\n")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive similarity analysis with threshold counting and detailed reports')
    parser.add_argument('files', nargs='*', 
                       help='JSON file paths (supports wildcards). If not provided, uses default file.')
    parser.add_argument('--thresholds', '-t', type=float, nargs='+', 
                       default=[0.647, 0.68, 0.7],
                       help='Similarity thresholds (default: 0.675 0.68 0.7)')
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='Generate detailed comprehensive report (default: simple report)')
    parser.add_argument('--output-csv', '-c', type=str,
                       help='Export detailed results to CSV file')
    parser.add_argument('--output-summary', '-s', type=str,
                       help='Export summary report to text file')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress console output')
    
    args = parser.parse_args()
    
    # Default file if none provided
    if not args.files:
        args.files = ["real_attack_results/pwws_vox1-O_b6_vox2_ft_lm_16_partial_batch_1.json"]
    
    # Expand wildcards
    all_files = []
    for pattern in args.files:
        if '*' in pattern or '?' in pattern:
            all_files.extend(glob.glob(pattern))
        else:
            all_files.append(pattern)
    
    if not all_files:
        print("No files found to process.")
        return
    
    # Sort thresholds and files
    thresholds = sorted(args.thresholds)
    all_files = sorted(all_files)
    
    # Analyze all files
    all_data = []
    for file_path in all_files:
        if os.path.exists(file_path):
            data = analyze_json_file(file_path)
            all_data.append(data)
        else:
            print(f"Warning: File not found: {file_path}")
    
    if not all_data:
        print("No files were successfully processed.")
        return
    # Generate summary statistics for detailed report
    summary_stats = generate_summary_stats(all_data) if args.detailed else None
    
    # Print appropriate report to console
    if not args.quiet:
        if args.detailed:
            print_detailed_report(all_data, thresholds, summary_stats)
        else:
            print_simple_report(all_data, thresholds)
    
    # Export to files
    if args.output_csv:
        export_to_csv(all_data, thresholds, args.output_csv)
        print(f"\n📊 Detailed results exported to: {args.output_csv}")
    
    if args.output_summary:
        export_summary_report(all_data, thresholds, summary_stats, args.output_summary)
        print(f"📄 Summary report exported to: {args.output_summary}")

if __name__ == "__main__":
    main() 