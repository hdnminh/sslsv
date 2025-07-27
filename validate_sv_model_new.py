import os
import sys
import pickle
import hashlib
import torch
from pathlib import Path
from dataclasses import dataclass, field

# Set up paths so we can import notebooks_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'notebooks'))

from notebooks.notebooks_utils import load_models, evaluate_models, create_metrics_df
from sslsv.evaluations.CosineSVEvaluation import CosineSVEvaluation, CosineSVEvaluationTaskConfig

def create_fake_audio_trials(
    fake_audio_base_path: str = "/media/volume/AudioUnlearnData1/sslsv/cache/original-fake-audios-05072025_16khz/F5TTSGenerator/seeded",
    real_audio_base_path: str = "/media/volume/AudioUnlearnData1/sslsv/data/voxceleb1",
    original_protocol: str = "data/voxceleb1_test_O",
    output_trials_file: str = "fake_audio_trials"
) -> str:
    """
    Create TTS attack trials by replicating VoxCeleb1 test protocol but replacing 
    target audio (third column) with TTS-generated fake audio
    
    Args:
        fake_audio_base_path: Path to fake audio files
        real_audio_base_path: Path to real VoxCeleb audio files  
        original_protocol: Path to original VoxCeleb1 test protocol
        output_trials_file: Output trial file name
        
    Returns:
        Path to created trials file
    """
    # Create mapping of original audio files to fake audio files
    fake_audio_mapping = {}
    
    print("Building fake audio mapping...")
    fake_speakers = os.listdir(fake_audio_base_path)
    
    for speaker_id in fake_speakers:
        speaker_fake_path = os.path.join(fake_audio_base_path, speaker_id)
        
        if not os.path.isdir(speaker_fake_path):
            continue
            
        # Get all video directories for this speaker
        video_dirs = os.listdir(speaker_fake_path)
        
        for video_id in video_dirs:
            video_fake_path = os.path.join(speaker_fake_path, video_id)
            
            if not os.path.isdir(video_fake_path):
                continue
                
            # Get fake audio files in this video directory
            fake_audio_files = [f for f in os.listdir(video_fake_path) if f.endswith('.wav')]
            
            for fake_audio_file in fake_audio_files:
                # Extract original audio filename (remove _seed16 suffix)
                original_audio_file = fake_audio_file.replace('_seed16.wav', '.wav')
                original_path = f"voxceleb1/{speaker_id}/{video_id}/{original_audio_file}"
                fake_path = f"fake_audio/{speaker_id}/{video_id}/{fake_audio_file}"
                
                fake_audio_mapping[original_path] = fake_path
    
    print(f"Found {len(fake_audio_mapping)} fake audio files")
    
    # Read original VoxCeleb1 test protocol and create fake version
    fake_trials = []
    trials_processed = 0
    trials_with_fake = 0
    trials_skipped = 0
    
    print(f"Processing original protocol: {original_protocol}")
    
    with open(original_protocol, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 3:
                continue
                
            label = parts[0]  # Keep original label (0 or 1)
            reference_audio = parts[1]  # Keep original reference audio
            target_audio = parts[2]  # This will be replaced with fake audio
            
            trials_processed += 1
            
            # Check if we have a fake version of the target audio
            if target_audio in fake_audio_mapping:
                fake_target_audio = fake_audio_mapping[target_audio]
                # Create new trial: same label, same reference, fake target
                fake_trial = f"{label} {reference_audio} {fake_target_audio}"
                fake_trials.append(fake_trial)
                trials_with_fake += 1
            else:
                trials_skipped += 1
    
    # Write fake trials to file
    with open(output_trials_file, 'w') as f:
        f.write('\n'.join(fake_trials))
    
    print(f"Created TTS attack protocol: {output_trials_file}")
    print(f"  - Original trials processed: {trials_processed}")
    print(f"  - Trials with fake audio: {trials_with_fake}")
    print(f"  - Trials skipped (no fake audio): {trials_skipped}")
    print(f"  - Coverage: {trials_with_fake/trials_processed*100:.1f}%")
    print(f"  - Format: <original_label> <original_reference> <fake_target>")
    print(f"  - This replicates VoxCeleb1_test_O but with fake target audio")
    
    return output_trials_file

def create_combined_trials(
    original_trials_file: str = "data/voxceleb1_test_O",
    fake_trials_file: str = "data/fake_audio_trials", 
    output_trials_file: str = "data/combined_trials"
) -> str:
    """
    Combine original real audio trials with fake audio trials into a single test set
    
    Args:
        original_trials_file: Path to original VoxCeleb1 trials
        fake_trials_file: Path to fake audio trials  
        output_trials_file: Path for combined trials output
        
    Returns:
        Path to the created combined trials file
    """
    try:
        combined_trials = []
        original_count = 0
        fake_count = 0
        
        # Read original trials
        print("Reading original VoxCeleb1 trials...")
        if os.path.exists(original_trials_file):
            with open(original_trials_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        combined_trials.append(line)
                        original_count += 1
        
        # Read fake audio trials
        print("Reading fake audio trials...")
        if os.path.exists(fake_trials_file):
            with open(fake_trials_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        combined_trials.append(line)
                        fake_count += 1
        
        # Write combined trials
        with open(output_trials_file, 'w') as f:
            f.write('\n'.join(combined_trials))
        
        print(f"Created combined trials file: {output_trials_file}")
        print(f"  - Original trials: {original_count}")
        print(f"  - Fake audio trials: {fake_count}")
        print(f"  - Total combined trials: {len(combined_trials)}")
        print(f"  - Format: mixed real-vs-real and real-vs-fake trials")
        
        return output_trials_file
        
    except Exception as e:
        print(f"Error creating combined trials: {e}")
        return None

class CachedCosineSVEvaluation(CosineSVEvaluation):
    """
    CosineSVEvaluation with embedding caching to speed up repeated evaluations
    """
    
    def __init__(self, cache_dir: str = "cache/embeddings", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a unique cache key based on available config attributes
        try:
            # Try different possible config structures
            if hasattr(self.config, 'model') and hasattr(self.config.model, 'name'):
                model_name = self.config.model.name
                enc_dim = getattr(self.config.model, 'enc_dim', 'default')
            elif hasattr(self.config, 'name'):
                model_name = self.config.name
                enc_dim = 'default'
            else:
                # Fallback to a generic identifier
                model_name = 'ssl_model'
                enc_dim = 'default'
            
            model_key = f"{model_name}_{enc_dim}"
            
            # Add dataset path if available
            if hasattr(self.config, 'dataset') and hasattr(self.config.dataset, 'base_path'):
                dataset_key = hashlib.md5(str(self.config.dataset.base_path).encode()).hexdigest()[:8]
                model_key += f"_{dataset_key}"
            
        except Exception as e:
            print(f"Warning: Could not create detailed cache key: {e}")
            # Ultimate fallback
            model_key = "ssl_model_default"
        
        self.cache_subdir = self.cache_dir / model_key
        self.cache_subdir.mkdir(exist_ok=True)
        
        print(f"Using embedding cache: {self.cache_subdir}")
        print(f"Config attributes: {[attr for attr in dir(self.config) if not attr.startswith('_')]}")
        
        # Store thresholds for later access
        self.eer_thresholds = {}
    
    def evaluate(self):
        """Override evaluate to capture EER thresholds"""
        # Call parent evaluation
        metrics = super().evaluate()
        
        # Extract EER thresholds for each trial
        from scipy.optimize import brentq
        from scipy.interpolate import interp1d
        from sklearn.metrics import roc_curve
        import numpy as np
        
        # Access the scores and targets that were computed during parent evaluation
        for trial_name in self.task_config.trials:
            try:
                # The parent class stores scores and targets after evaluation
                scores_attr = None
                targets_attr = None
                
                # Check for scores and targets attributes
                if hasattr(self, 'scores') and hasattr(self, 'targets'):
                    scores_attr = self.scores
                    targets_attr = self.targets
                
                if scores_attr is not None and targets_attr is not None:
                    scores = np.array(scores_attr)
                    targets = np.array(targets_attr)
                    
                    # Compute ROC curve
                    fpr, tpr, thresholds = roc_curve(targets, scores, pos_label=1)
                    
                    # Find EER threshold using interpolation
                    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
                    eer_threshold = interp1d(fpr, thresholds)(eer)
                    
                    # Ensure scalar output
                    if hasattr(eer_threshold, 'item'):
                        eer_threshold = float(eer_threshold.item())
                    elif isinstance(eer_threshold, np.ndarray):
                        eer_threshold = float(eer_threshold.squeeze())
                    else:
                        eer_threshold = float(eer_threshold)
                    
                    self.eer_thresholds[trial_name] = eer_threshold
                    
                else:
                    self.eer_thresholds[trial_name] = None
                    
            except Exception as e:
                print(f"Warning: Could not compute EER threshold for {trial_name}: {e}")
                self.eer_thresholds[trial_name] = None
        
        return metrics
    
    def _compute_eer_threshold_fallback(self, trial_name):
        """Fallback method to compute EER threshold by examining the evaluation internals"""
        # This method is kept for potential future use but not currently needed
        self.eer_thresholds[trial_name] = None
    
    def calculate_performance_at_threshold(self, trial_name, threshold):
        """Calculate FAR, FRR, and accuracy at a specific threshold"""
        try:
            import numpy as np
            
            if hasattr(self, 'scores') and hasattr(self, 'targets'):
                scores = np.array(self.scores)
                targets = np.array(self.targets)
                
                # Apply threshold: scores >= threshold are classified as same-speaker (1)
                predictions = (scores >= threshold).astype(int)
                
                # Calculate confusion matrix components
                true_positives = np.sum((targets == 1) & (predictions == 1))
                false_positives = np.sum((targets == 0) & (predictions == 1))
                true_negatives = np.sum((targets == 0) & (predictions == 0))
                false_negatives = np.sum((targets == 1) & (predictions == 0))
                
                # Calculate rates
                total_positives = np.sum(targets == 1)
                total_negatives = np.sum(targets == 0)
                
                if total_positives > 0:
                    true_positive_rate = true_positives / total_positives  # 1 - FRR
                    false_rejection_rate = false_negatives / total_positives  # FRR
                else:
                    true_positive_rate = 0.0
                    false_rejection_rate = 0.0
                
                if total_negatives > 0:
                    false_acceptance_rate = false_positives / total_negatives  # FAR
                    true_negative_rate = true_negatives / total_negatives  # 1 - FAR
                else:
                    false_acceptance_rate = 0.0
                    true_negative_rate = 0.0
                
                accuracy = (true_positives + true_negatives) / len(targets)
                
                # Calculate error rate at this threshold
                error_rate = (false_positives + false_negatives) / len(targets)
                
                return {
                    'threshold': threshold,
                    'far': false_acceptance_rate * 100,  # Convert to percentage
                    'frr': false_rejection_rate * 100,   # Convert to percentage
                    'accuracy': accuracy * 100,
                    'error_rate': error_rate * 100,
                    'true_positives': true_positives,
                    'false_positives': false_positives,
                    'true_negatives': true_negatives,
                    'false_negatives': false_negatives,
                    'total_positives': total_positives,
                    'total_negatives': total_negatives
                }
            else:
                return None
                
        except Exception as e:
            print(f"Error calculating performance at threshold {threshold}: {e}")
            return None
    
    def _get_cache_path(self, audio_path: str) -> Path:
        """Get cache file path for an audio file"""
        # Create a safe filename from the audio path
        safe_name = audio_path.replace('/', '_').replace('\\', '_')
        cache_file = f"{safe_name}.pkl"
        return self.cache_subdir / cache_file
    
    def _load_cached_embedding(self, audio_path: str):
        """Load embedding from cache if it exists"""
        cache_path = self._get_cache_path(audio_path)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Failed to load cache for {audio_path}: {e}")
                return None
        return None
    
    def _save_embedding_to_cache(self, audio_path: str, embedding):
        """Save embedding to cache"""
        cache_path = self._get_cache_path(audio_path)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            print(f"Failed to save cache for {audio_path}: {e}")
    
    def _extract_test_embeddings(self, trial_names):
        """Override to add caching functionality while maintaining parent class compatibility"""
        if not hasattr(self, 'test_embeddings'):
            self.test_embeddings = {}
        
        for trial_name in trial_names:
            print(f"Processing trial: {trial_name}")
            
            # Get trial file path
            trial_file_path = self.config.dataset.base_path / trial_name
            
            # Parse trial file and extract unique files
            files = set()
            with open(trial_file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        files.add(parts[1])  # reference audio
                        files.add(parts[2])  # test audio
            
            files = list(files)
            
            # Check cache for each file
            cached_count = 0
            to_compute = []
            
            for file_path in files:
                if file_path not in self.test_embeddings:  # Check if already in flat structure
                    cached_embedding = self._load_cached_embedding(file_path)
                    if cached_embedding is not None:
                        self.test_embeddings[file_path] = cached_embedding  # Store in flat structure
                        cached_count += 1
                    else:
                        to_compute.append(file_path)
                else:
                    cached_count += 1  # Already computed in previous trial
            
            print(f"  - Found {cached_count} cached embeddings")
            print(f"  - Need to compute {len(to_compute)} new embeddings")
            
            # For missing embeddings, compute and store in flat structure
            if to_compute:
                print(f"  - Extracting {len(to_compute)} embeddings...")
                
                for i, file_path in enumerate(to_compute):
                    if i % 100 == 0:
                        print(f"    Progress: {i}/{len(to_compute)}")
                    
                    try:
                        # Call parent's embedding extraction logic
                        embedding = self._extract_single_embedding(file_path)
                        self.test_embeddings[file_path] = embedding  # Store in flat structure
                        
                        # Save to cache
                        self._save_embedding_to_cache(file_path, embedding)
                        
                    except Exception as e:
                        print(f"    Failed to extract embedding for {file_path}: {e}")
                        # Skip this file instead of using fallback
                        continue
            
            print(f"  - Total embeddings loaded for {trial_name}: {len([f for f in files if f in self.test_embeddings])}")
            print(f"  - Total embeddings in memory: {len(self.test_embeddings)}")
    
    def _extract_single_embedding(self, file_path):
        """Extract embedding for a single file using parent class logic"""
        # Load audio
        audio_full_path = self.config.dataset.base_path / file_path
        import librosa
        
        audio, sr = librosa.load(audio_full_path, sr=16000)
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        
        # Extract embedding using the same logic as parent class
        with torch.no_grad():
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            audio_tensor = audio_tensor.to(self.device)
            
            # Apply the model to get raw embeddings
            embedding = self.model(audio_tensor)
            
            # Ensure the embedding has the right shape for frame-level processing
            # The SSL framework expects embeddings to be [num_frames, embedding_dim]
            if embedding.dim() == 2:
                # Already has batch dimension, just normalize
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
            elif embedding.dim() == 1:
                # Add frame dimension: [embedding_dim] -> [1, embedding_dim]
                embedding = embedding.unsqueeze(0)
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
            else:
                # Handle other cases by flattening and adding frame dimension
                embedding = embedding.view(-1, embedding.shape[-1])
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
            
            # Move to CPU for storage
            embedding = embedding.cpu()
        
        return embedding


def calculate_cross_threshold_performance(original_evaluator, fake_evaluator, original_threshold, fake_threshold):
    """
    Calculate performance metrics using cross-thresholds:
    - Original data performance at TTS attack threshold
    - TTS data performance at original threshold
    """
    results = {}
    
    try:
        # Calculate original data performance at TTS attack threshold
        if original_evaluator and fake_threshold is not None:
            original_at_fake_thresh = original_evaluator.calculate_performance_at_threshold(
                'voxceleb1_test_O', fake_threshold
            )
            if original_at_fake_thresh:
                results['original_at_fake_threshold'] = original_at_fake_thresh
        
        # Calculate TTS data performance at original threshold  
        if fake_evaluator and original_threshold is not None:
            fake_at_original_thresh = fake_evaluator.calculate_performance_at_threshold(
                'fake_audio_trials', original_threshold
            )
            if fake_at_original_thresh:
                results['fake_at_original_threshold'] = fake_at_original_thresh
        
        # Also get native threshold performances for comparison
        if original_evaluator and original_threshold is not None:
            original_at_original_thresh = original_evaluator.calculate_performance_at_threshold(
                'voxceleb1_test_O', original_threshold
            )
            if original_at_original_thresh:
                results['original_at_original_threshold'] = original_at_original_thresh
        
        if fake_evaluator and fake_threshold is not None:
            fake_at_fake_thresh = fake_evaluator.calculate_performance_at_threshold(
                'fake_audio_trials', fake_threshold
            )
            if fake_at_fake_thresh:
                results['fake_at_fake_threshold'] = fake_at_fake_thresh
                
    except Exception as e:
        print(f"Error in cross-threshold analysis: {e}")
    
    return results


def print_cross_threshold_analysis(results):
    """Print formatted cross-threshold analysis results"""
    if not results:
        print("❌ No cross-threshold results available")
        return
    
    print("\n" + "=" * 60)
    print("CROSS-THRESHOLD ANALYSIS")
    print("=" * 60)
    
    # Original data at TTS threshold
    if 'original_at_fake_threshold' in results:
        orig_fake_thresh = results['original_at_fake_threshold']
        print(f"📊 ORIGINAL DATA AT TTS ATTACK THRESHOLD ({orig_fake_thresh['threshold']:.6f}):")
        print(f"   FAR: {orig_fake_thresh['far']:.2f}%")
        print(f"   FRR: {orig_fake_thresh['frr']:.2f}%")
        print(f"   Error Rate: {orig_fake_thresh['error_rate']:.2f}%")
        print(f"   Accuracy: {orig_fake_thresh['accuracy']:.2f}%")
    
    # TTS data at original threshold
    if 'fake_at_original_threshold' in results:
        fake_orig_thresh = results['fake_at_original_threshold']
        print(f"\n📊 TTS ATTACK DATA AT ORIGINAL THRESHOLD ({fake_orig_thresh['threshold']:.6f}):")
        print(f"   FAR: {fake_orig_thresh['far']:.2f}%")
        print(f"   FRR: {fake_orig_thresh['frr']:.2f}%")
        print(f"   Error Rate: {fake_orig_thresh['error_rate']:.2f}%")
        print(f"   Accuracy: {fake_orig_thresh['accuracy']:.2f}%")
    
    # Comparison analysis
    if 'original_at_fake_threshold' in results and 'original_at_original_threshold' in results:
        orig_fake = results['original_at_fake_threshold']
        orig_orig = results['original_at_original_threshold']
        
        print(f"\n🔍 THRESHOLD IMPACT ON ORIGINAL DATA:")
        print(f"   At original threshold ({orig_orig['threshold']:.6f}): {orig_orig['error_rate']:.2f}% error")
        print(f"   At TTS attack threshold ({orig_fake['threshold']:.6f}): {orig_fake['error_rate']:.2f}% error")
        
        error_change = orig_fake['error_rate'] - orig_orig['error_rate']
        if abs(error_change) < 0.1:
            print(f"   ➤ Minimal impact: {error_change:+.2f}% change")
        elif error_change > 0:
            print(f"   ➤ TTS threshold degrades performance: {error_change:+.2f}% increase in error")
        else:
            print(f"   ➤ TTS threshold improves performance: {error_change:+.2f}% decrease in error")
    
    # Attack effectiveness analysis
    if 'fake_at_original_threshold' in results and 'fake_at_fake_threshold' in results:
        fake_orig = results['fake_at_original_threshold']
        fake_fake = results['fake_at_fake_threshold']
        
        print(f"\n🎯 ATTACK OPTIMIZATION ANALYSIS:")
        print(f"   TTS at original threshold: {fake_orig['error_rate']:.2f}% error")
        print(f"   TTS at optimized threshold: {fake_fake['error_rate']:.2f}% error")
        
        optimization_gain = fake_fake['error_rate'] - fake_orig['error_rate']
        if optimization_gain > 1.0:
            print(f"   ➤ Attack benefits significantly from threshold optimization: {optimization_gain:+.2f}%")
        elif optimization_gain > 0.1:
            print(f"   ➤ Attack benefits from threshold optimization: {optimization_gain:+.2f}%")
        else:
            print(f"   ➤ Attack is robust across thresholds: {optimization_gain:+.2f}%")


if __name__ == "__main__":
    # Paths to config and checkpoint
    config = "models/tests/simclr/simclr_e-ecapa/ssps_kmeans_25k_uni-1/config.yml"
    checkpoint = "models/tests/simclr/simclr_e-ecapa/ssps_kmeans_25k_uni-1/checkpoints/model_avg.pt"

    # Load model from config and checkpoint
    models = load_models([config], checkpoint_name=os.path.basename(checkpoint))

    print("=" * 60)
    print("EVALUATING ORIGINAL VOXCELEB1 DATA")
    print("=" * 60)
    
    # Original VoxCeleb evaluation
    eval_task_config = CosineSVEvaluationTaskConfig(
        __type__="sv_cosine",
        frame_length=None,
        num_frames=1,
        trials=['voxceleb1_test_O'],
        metrics=['eer', 'mindcf']
    )

    # Evaluate original data and capture evaluation object
    original_evaluator = None
    for model_name, model_entry in models.items():
        evaluator = CachedCosineSVEvaluation(
            model=model_entry.model,
            config=model_entry.config,
            task_config=eval_task_config,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=True
        )
        metrics = evaluator.evaluate()
        model_entry.metrics = metrics
        original_evaluator = evaluator  # Store for threshold access
        break  # Only need one model

    # Print original metrics
    for model_name, model_entry in models.items():
        print(f"Model: {model_name}, metric keys: {list(getattr(model_entry, 'metrics', {}).keys())}")
    
    # Print original results directly instead of using create_metrics_df
    print("Original VoxCeleb1 Results:")
    for model_name, model_entry in models.items():
        metrics = getattr(model_entry, 'metrics', {})
        original_eer = metrics.get('voxceleb1_test_O/eer', 'N/A')
        original_mindcf = metrics.get('voxceleb1_test_O/mindcf', 'N/A')
        print(f"  {model_name}:")
        print(f"    EER: {original_eer}")
        print(f"    minDCF: {original_mindcf}")
    
    # Extract original EER threshold
    original_eer_threshold = None
    if original_evaluator and hasattr(original_evaluator, 'eer_thresholds'):
        if 'voxceleb1_test_O' in original_evaluator.eer_thresholds:
            original_eer_threshold = original_evaluator.eer_thresholds['voxceleb1_test_O']
            print(f"Original VoxCeleb1 EER Threshold: {original_eer_threshold:.6f}")
    
    # Save original metrics before they get overwritten
    original_metrics = {}
    for model_name, model_entry in models.items():
        original_metrics[model_name] = model_entry.metrics.copy()
    
    print("\n" + "=" * 60)
    print("EVALUATING TTS-GENERATED FAKE AUDIO")
    print("=" * 60)
    
    # Create symlink for fake audio so SSL framework can find it
    fake_audio_base_path = "/media/volume/AudioUnlearnData1/sslsv/cache/original-fake-audios-05072025_16khz/F5TTSGenerator/seeded"
    fake_audio_symlink = "data/fake_audio"
    
    # Remove existing symlink if it exists
    if os.path.exists(fake_audio_symlink):
        os.remove(fake_audio_symlink)
    
    # Create symlink to fake audio directory
    os.symlink(fake_audio_base_path, fake_audio_symlink)
    
    # Create fake audio trials in the data directory where SSL framework expects it
    trials_file = create_fake_audio_trials(
        fake_audio_base_path=fake_audio_base_path,
        real_audio_base_path="/media/volume/AudioUnlearnData1/sslsv/data/voxceleb1",
        original_protocol="data/voxceleb1_test_O",
        output_trials_file="data/fake_audio_trials"
    )
    
    # Evaluate fake audio using standard SSL framework
    fake_eval_task_config = CosineSVEvaluationTaskConfig(
        __type__="sv_cosine",
        frame_length=None,
        num_frames=1,
        trials=['fake_audio_trials'],
        metrics=['eer', 'mindcf']
    )
    
    # Use the standard evaluate_models function with cached evaluation and capture evaluator
    fake_evaluator = None
    for model_name, model_entry in models.items():
        evaluator = CachedCosineSVEvaluation(
            model=model_entry.model,
            config=model_entry.config,
            task_config=fake_eval_task_config,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=True
        )
        metrics = evaluator.evaluate()
        model_entry.metrics = metrics
        fake_evaluator = evaluator  # Store for threshold access
        break  # Only need one model
    
    # Print TTS attack results directly instead of using create_metrics_df
    print("TTS Attack Results (Fake Audio Only):")
    for model_name, model_entry in models.items():
        metrics = getattr(model_entry, 'metrics', {})
        fake_eer = metrics.get('fake_audio_trials/eer', 'N/A')
        fake_mindcf = metrics.get('fake_audio_trials/mindcf', 'N/A')
        print(f"  {model_name}:")
        print(f"    EER: {fake_eer}")
        print(f"    minDCF: {fake_mindcf}")
    
    # Extract fake audio EER threshold
    fake_eer_threshold = None
    if fake_evaluator and hasattr(fake_evaluator, 'eer_thresholds'):
        if 'fake_audio_trials' in fake_evaluator.eer_thresholds:
            fake_eer_threshold = fake_evaluator.eer_thresholds['fake_audio_trials']
            print(f"TTS Attack EER Threshold: {fake_eer_threshold:.6f}")
    
    print("\n" + "=" * 60)
    print("TTS ATTACK ANALYSIS")
    print("=" * 60)
    
    # Extract EER values for TTS attack analysis
    try:
        # Get fake audio metrics directly from model entry
        model_entry = list(models.values())[0]  # Get first (and likely only) model
        metrics = getattr(model_entry, 'metrics', {})
        fake_eer = metrics.get('fake_audio_trials/eer', None)
        fake_mindcf = metrics.get('fake_audio_trials/mindcf', None)
        
        # Get original metrics from saved data
        model_entry_first = list(models.values())[0]  # Get first (and likely only) model
        original_metrics_saved = original_metrics[list(models.keys())[0]]
        original_eer = original_metrics_saved.get('voxceleb1_test_O/eer', None)
        original_mindcf = original_metrics_saved.get('voxceleb1_test_O/mindcf', None)
        
        if fake_eer is None or fake_mindcf is None:
            print("❌ Could not extract fake audio metrics")
        elif original_eer is None or original_mindcf is None:
            print("❌ Could not extract original audio metrics")
        else:
            print(f"📊 METRICS COMPARISON:")
            print(f"   Original VoxCeleb1 EER:     {original_eer:.4f}%")
            print(f"   TTS Attack EER:             {fake_eer:.4f}%")
            print(f"   Original VoxCeleb1 minDCF:  {original_mindcf:.4f}")
            print(f"   TTS Attack minDCF:          {fake_mindcf:.4f}")
            
            print(f"\n🎯 EER THRESHOLDS (DECISION BOUNDARIES):")
            if original_eer_threshold is not None:
                print(f"   Original VoxCeleb1 Threshold: {original_eer_threshold:.6f}")
            if fake_eer_threshold is not None:
                print(f"   TTS Attack Threshold:         {fake_eer_threshold:.6f}")
            if original_eer_threshold is not None and fake_eer_threshold is not None:
                threshold_diff = abs(fake_eer_threshold - original_eer_threshold)
                print(f"   Threshold Difference:         {threshold_diff:.6f}")
                
                if threshold_diff < 0.01:
                    print(f"   ➤ Similar thresholds suggest consistent model behavior")
                else:
                    print(f"   ➤ Different thresholds suggest TTS affects decision boundary")
            
            print(f"\n🎯 TTS ATTACK EFFECTIVENESS:")
            if fake_eer >= 99.0:
                print("   ❌ TTS ATTACK COMPLETELY INEFFECTIVE")
                print("   ➤ SSL model perfectly distinguishes real vs TTS audio")
                print("   ➤ EER ~100% means model never confuses TTS for real speaker")
            elif fake_eer > original_eer * 2:
                print("   ⚠️  TTS ATTACK MOSTLY INEFFECTIVE") 
                print(f"   ➤ SSL model easily detects TTS audio (EER {fake_eer:.1f}% vs {original_eer:.1f}%)")
            elif fake_eer > original_eer:
                print("   ✅ TTS ATTACK SOMEWHAT EFFECTIVE")
                effectiveness = ((fake_eer - original_eer) / original_eer) * 100
                print(f"   ➤ Attack increases error rate by {effectiveness:.1f}%")
            else:
                print("   🔥 TTS ATTACK HIGHLY EFFECTIVE")
                print("   ➤ TTS audio fools SSL model as well as real audio")
                
            print(f"\n📈 INTERPRETATION:")
            print(f"   • EER {fake_eer:.1f}%: {fake_eer:.1f}% of TTS attacks are incorrectly verified")
            print(f"   • minDCF {fake_mindcf:.4f}: Detection cost when optimally configured")
            
            if fake_eer >= 50.0:
                print(f"   • High EER suggests strong SSL model resistance to TTS spoofing")
            else:
                print(f"   • Low EER suggests SSL model vulnerability to TTS spoofing")

    except Exception as e:
        print(f"Could not analyze TTS attack results: {e}")
        print("Available metrics:", [list(getattr(model_entry, 'metrics', {}).keys()) for model_entry in models.values()])

    # Perform cross-threshold analysis
    cross_results = calculate_cross_threshold_performance(
        original_evaluator, fake_evaluator, original_eer_threshold, fake_eer_threshold
    )
    print_cross_threshold_analysis(cross_results)

    print("\n" + "=" * 60)
    print("EVALUATING COMBINED DATASET (REAL + FAKE)")
    print("=" * 60)

    # Create combined trials file
    combined_trials_file = create_combined_trials(
        original_trials_file="data/voxceleb1_test_O",
        fake_trials_file="data/fake_audio_trials",
        output_trials_file="data/combined_trials"
    )

    if combined_trials_file:
        # Evaluate combined dataset to get mixed-optimized threshold
        combined_eval_task_config = CosineSVEvaluationTaskConfig(
            __type__="sv_cosine",
            frame_length=None,
            num_frames=1,
            trials=['combined_trials'],
            metrics=['eer', 'mindcf']
        )

        # Use cached evaluation for combined dataset
        combined_evaluator = None
        for model_name, model_entry in models.items():
            evaluator = CachedCosineSVEvaluation(
                model=model_entry.model,
                config=model_entry.config,
                task_config=combined_eval_task_config,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                verbose=True
            )
            metrics = evaluator.evaluate()
            combined_evaluator = evaluator  # Store for threshold access
            break  # Only need one model

        # Display combined results
        print("Combined Dataset Results (Real + Fake):")
        if combined_evaluator and hasattr(combined_evaluator, 'eer_thresholds'):
            combined_eer_threshold = combined_evaluator.eer_thresholds.get('combined_trials', None)
            if combined_eer_threshold is not None:
                print(f"  Mixed-Optimized EER Threshold: {combined_eer_threshold:.6f}")
                
                # Get combined dataset metrics
                combined_metrics = getattr(combined_evaluator, 'scores', None)
                if combined_metrics is not None:
                    # Calculate EER and minDCF for display
                    combined_performance = combined_evaluator.calculate_performance_at_threshold(
                        'combined_trials', combined_eer_threshold
                    )
                    if combined_performance:
                        print(f"  Combined Dataset EER: {combined_performance['error_rate']:.4f}%")
                        print(f"  Combined Dataset FAR: {combined_performance['far']:.2f}%")
                        print(f"  Combined Dataset FRR: {combined_performance['frr']:.2f}%")
                        print(f"  Combined Dataset Accuracy: {combined_performance['accuracy']:.2f}%")

                print("\n" + "=" * 60)
                print("RE-EVALUATING ORIGINAL DATA WITH MIXED-OPTIMIZED THRESHOLD")
                print("=" * 60)

                # Re-evaluate original data using the mixed-optimized threshold
                original_at_mixed_threshold = original_evaluator.calculate_performance_at_threshold(
                    'voxceleb1_test_O', combined_eer_threshold
                )

                if original_at_mixed_threshold:
                    print(f"📊 ORIGINAL DATA AT MIXED-OPTIMIZED THRESHOLD ({combined_eer_threshold:.6f}):")
                    print(f"   FAR: {original_at_mixed_threshold['far']:.2f}%")
                    print(f"   FRR: {original_at_mixed_threshold['frr']:.2f}%")
                    print(f"   Error Rate: {original_at_mixed_threshold['error_rate']:.2f}%")
                    print(f"   Accuracy: {original_at_mixed_threshold['accuracy']:.2f}%")

                    print(f"\n🔍 THRESHOLD COMPARISON ON ORIGINAL DATA:")
                    print(f"   Original-optimized threshold ({original_eer_threshold:.6f}): {original_metrics_saved.get('voxceleb1_test_O/eer', 'N/A'):.2f}% EER")
                    print(f"   TTS-optimized threshold ({fake_eer_threshold:.6f}): 3.20% error rate")
                    print(f"   Mixed-optimized threshold ({combined_eer_threshold:.6f}): {original_at_mixed_threshold['error_rate']:.2f}% error rate")

                    # Analyze the impact
                    original_eer_value = float(original_metrics_saved.get('voxceleb1_test_O/eer', 0))
                    mixed_error_change = original_at_mixed_threshold['error_rate'] - original_eer_value
                    
                    print(f"\n📈 MIXED-THRESHOLD IMPACT:")
                    if abs(mixed_error_change) < 0.1:
                        print(f"   ➤ Minimal impact: {mixed_error_change:+.2f}% change from original")
                    elif mixed_error_change > 0:
                        print(f"   ➤ Mixed threshold slightly degrades performance: {mixed_error_change:+.2f}% increase")
                    else:
                        print(f"   ➤ Mixed threshold slightly improves performance: {mixed_error_change:+.2f}% decrease")
                    
                    print(f"   ➤ Mixed-optimized threshold balances real and fake detection")
            else:
                print("  Could not compute mixed-optimized EER threshold")
        else:
            print("  Could not evaluate combined dataset")
    else:
        print("  Failed to create combined trials file")




