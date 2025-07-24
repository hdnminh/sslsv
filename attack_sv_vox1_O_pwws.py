#!/usr/bin/env python3
"""
Speech Verification Attack Script
Performs adversarial attacks on speech verification systems using text-to-speech
"""

import os
import gc
import json
import math
import time
import torch
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm
import csv
from loguru import logger
import datetime
import librosa


from evaluate import load
from utils import load_dataset, get_attack_log_summary
from tts_utils import TTS_MODELS_REGISTER, TTSGenerator

from textattack import Attacker
from textattack.datasets import Dataset
from textattack.models.wrappers import ModelWrapper
from textattack.constraints.semantics.sentence_encoders.sentence_bert import SBERT
from textattack.attack_recipes import PWWSRen2019, TextFoolerJin2019, BERTAttackLi2020, BAEGarg2019
from textattack.constraints.semantics.sentence_encoders.universal_sentence_encoder import UniversalSentenceEncoder
from textattack.goal_functions import GoalFunction
from textattack.goal_functions.classification import ClassificationGoalFunction
from textattack.search_methods import SearchMethod
from textattack.transformations import Transformation
from textattack.shared import AttackedText
from textattack.attack import Attack

from similarity_maximization_goal import SimilarityMaximizationGoal


# --- Pre-download SBERT model to avoid repeated requests and rate limits ---
try:
    from sentence_transformers import SentenceTransformer
    print("Pre-downloading SBERT model 'sentence-transformers/all-MiniLM-L6-v2' if not already cached...")
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("SBERT model pre-download complete.")
except Exception as e:
    print(f"Could not pre-download SBERT model: {e}")
# --- End pre-download step ---

# Note: Directories will be created later once args are parsed

# Set environment variable for CUDA deterministic behavior
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Set deterministic behavior for PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)


ATTACKER_MAP = {
    "pwws": PWWSRen2019,
    # "text_fooler": TextFoolerJin2019,
    # "bae": BAEGarg2019,
    # "bert_attack": BERTAttackLi2020,
}


def setup_directories(args):
    """Create all necessary directories for the experiment"""
    # Create output directory and its subdirectories
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)
    
    return log_dir

def setup_logging(log_dir, tts, attacker_name, seed_value):
    """Configure loguru with proper log directory"""
    logger.remove()  # Remove default handler
    logger.add(
        os.path.join(log_dir, f"real_attack_{tts}_attacker_{attacker_name}_seed_{seed_value}.log"),
        format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
        level="INFO",
        rotation="10 MB"
    )
    logger.add(
        lambda msg: print(msg, end=""),  # Console handler
        format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}\n",
        level="INFO"
    )

class SpeechVerificationModel:
    """Wrapper for ReDimNet speech verification models"""
    
    def __init__(self, model_name: str = 'b0', dataset: str = 'vox2', train_type: str = 'ft_lm'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.dataset = dataset
        self.train_type = train_type
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load pretrained ReDimNet model"""
        try:
            logger.info(f"Loading ReDimNet model: {self.model_name}, dataset: {self.dataset}, train_type: {self.train_type}")
            self.model = torch.hub.load('IDRnD/ReDimNet', 'ReDimNet', 
                                      model_name=self.model_name, 
                                      train_type=self.train_type, 
                                      dataset=self.dataset,
                                      trust_repo=True)
            self.model.to(self.device)  # type: ignore
            self.model.eval()  # type: ignore
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def extract_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract embedding from audio using the model"""
        try:
            with torch.no_grad():
                # Add batch dimension
                audio_batch = audio.unsqueeze(0).to(self.device)
                
                # Get embedding
                embedding = self.model(audio_batch)  # type: ignore
                
                # Normalize embedding
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                
                return embedding.squeeze(0).cpu()
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            return torch.tensor([], dtype=torch.float32)
    
    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between two embeddings"""
        return (torch.dot(emb1, emb2).item() + 1) / 2  # Convert from [-1, 1] to [0, 1]

class SpeechVerificationPipelineWrapper(ModelWrapper):
    """Wrapper for speech verification pipeline to work with TextAttack"""

    def __init__(
        self, 
        sv_model: SpeechVerificationModel, 
        tts: TTSGenerator, 
        reference_audio_path: str, 
        target_audio_path: str = '', 
        target_audio_rel_path: str = '', 
        target_transcript: str = '', 
        device=None, 
        use_seeding: bool = True, 
        use_cache: bool = True,
        cache_dir: str = "",
        seed_value: int = 16,
        original_fake_audios_path: str = ""
    ):
        self.sv_model = sv_model
        self.tts = tts
        self.reference_audio_path = reference_audio_path
        self.target_audio_path = target_audio_path
        self.target_audio_rel_path = target_audio_rel_path
        self.target_transcript = target_transcript
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_seeding = use_seeding
        self.seed_value = seed_value
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.tts_cnt = 0
        self.start_time = time.time()
        self.original_fake_audios_path = original_fake_audios_path
        # Load reference embedding
        self.reference_embedding = self._load_reference_embedding()
        
    def _load_reference_embedding(self) -> torch.Tensor:
        """Load and extract embedding from reference audio"""
        try:
            
            # Load reference audio
            audio, sr = librosa.load(self.reference_audio_path, sr=None)
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
            
            # Extract embedding
            embedding = self.sv_model.extract_embedding(audio_tensor)
            if embedding is None:
                raise ValueError("Failed to extract reference embedding")
            
            logger.info(f"Reference embedding loaded from {self.reference_audio_path}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to load reference embedding: {e}")
            raise
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio file and convert to tensor"""
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            return torch.tensor(audio, dtype=torch.float32)
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            return torch.tensor([], dtype=torch.float32)
    
    def reset_stats(self):
        """Reset timing and counter statistics"""
        self.start_time = time.time()
        self.tts_cnt = 0

    def get_time_cost(self):
        """Get formatted execution time"""
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        
        return f"{hours}h:{minutes}m:{seconds:.2f}s"
    
    def _set_deterministic_seed(self, seed_value: int):
        """Set deterministic seed for all random number generators"""
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
        
        import random
        import numpy as np
        random.seed(seed_value)
        np.random.seed(seed_value)

    @property
    def model(self):
        # Return self or the underlying SV model, as needed for TextAttack compatibility
        return self.sv_model

    def __call__(self, text_inputs):
        """
        Main inference method for TextAttack compatibility.
        
        Now uses TTSGenerator's enhanced caching - much simpler!
        
        Audio Caching Strategy:
        - Original Transcripts: Attempts to reuse pre-generated fake audio from 
          cache/original-fake-audios/{TTS_MODEL}/{seeded|unseeded}/{speaker_id}/{video_id}/{audio_file}_seed42.wav
          If not found, skips processing and returns neutral score [0.5, 0.5]
        - Perturbed Transcripts: Uses hash-based caching in 
          cache/speech-verification/{TTS_MODEL}/{seeded|unseeded}/{hash}_seed42.wav
          Generates new audio if not in cache
        
        Voice Cloning:
        - For CoquiTTS/F5TTS: Uses target_audio_path for voice cloning with proper reference audio/text
        - For KokoroTTS/OpenAI: Uses predefined voices (no voice cloning support)
        
        Seeding:
        - When use_seeding=True: Sets deterministic seeds (torch, random, numpy) to seed=42
        - Affects both audio generation and filename (_seed42 suffix)
        
        Returns:
        - numpy array of shape (N, 2) where each row is [match_prob, non_match_prob]
        - match_prob: cosine similarity between generated and reference audio (range [0, 1])
        - non_match_prob: 1 - match_prob
        - [0.5, 0.5]: neutral score for skipped samples or failures
        
        This format allows TextAttack to maximize similarity by maximizing the first element.
        
        """
        self.tts_cnt += len(text_inputs)
        outputs = []
        
        # Set deterministic seed if enabled
        if self.use_seeding:
            self._set_deterministic_seed(self.seed_value)
        
        # Use TTSGenerator's enhanced caching with speaker-based structure
        audio_files = self.tts(
            text_inputs,
            cache_mode='speaker_based', #This also applies loading the original fake audio generated before to ensure the fairness and consistency of the attack
            original_fake_audios_path=self.original_fake_audios_path, # The original fake audio files generated before the real attack lie  in this path
            use_seeding=self.use_seeding,
            seed_value=self.seed_value,
            target_transcript=self.target_transcript,
            target_audio_rel_path=self.target_audio_rel_path,
            target_audio_path=self.target_audio_path,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir
        )
        
        # Process each generated audio file
        for idx, (transcript, audio_file) in enumerate(zip(text_inputs, audio_files)):
            if audio_file is None:
                # Sample was skipped (e.g., original fake audio not found)
                outputs.append([0.5, 0.5])
                continue
                
            # Load the generated audio
            audio_tensor = self._load_audio(audio_file)
            if audio_tensor is None:
                outputs.append([0.5, 0.5])
                continue
                
            # Extract embedding and compute similarity
            embedding = self.sv_model.extract_embedding(audio_tensor)
            if embedding is None:
                outputs.append([0.5, 0.5])
                continue
                
            similarity = self.sv_model.compute_similarity(self.reference_embedding, embedding)
            match_prob = similarity
            non_match_prob = 1 - match_prob
            outputs.append([match_prob, non_match_prob])
            
        return np.array(outputs)

def save_partial_results(
    batch_results, 
    attacker_name, 
    sv_model, 
    tts, 
    file_name, 
    batch_num, 
    total_batches, 
    batch_start_time=None,
    ):
    """Save partial results after processing a batch"""
    logger.info(f"Saving partial results for {attacker_name} - batch {batch_num}/{total_batches}")
    
    # Create partial filename
    partial_file_name = file_name.replace('.json', f'_partial_batch_{batch_num}.json')
    
    # Calculate batch timing if provided
    batch_metadata = {
        "batch_number": batch_num,
        "total_batches": total_batches,
        "results_in_batch": len(batch_results)
    }
    
    if batch_start_time:
        batch_end_time = datetime.datetime.now()
        batch_runtime = (batch_end_time - batch_start_time).total_seconds()
        batch_metadata.update({
            "batch_start_time": batch_start_time.isoformat(),
            "batch_end_time": batch_end_time.isoformat(),
            "batch_runtime_seconds": batch_runtime,
            "batch_runtime_formatted": format_runtime(batch_runtime)
        })
    
    data = {
        "model_name": format_model_name(sv_model),
        "TTS_model": tts.__class__.__name__,
        "attacker": attacker_name,
        "attack_strategy": "maximize_similarity",
        "batch_info": batch_metadata,
        "results": batch_results,
    }

    with open(partial_file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    logger.info(f"Partial results saved to {partial_file_name}")

def save_final_consolidated_results(
    all_results, 
    attacker_name, 
    sv_model, 
    tts, 
    file_name, 
    total_tts_cnt, 
    total_time_cost, 
    start_timestamp=None, 
    end_timestamp=None, 
    total_runtime=None, 
    log_summary=None,
    ):
    """Save final consolidated results from all batches"""
    logger.info(f"Saving final consolidated results for {attacker_name} - total results: {len(all_results)}")
    
    data = {
        "log_summary": log_summary,
        "model_name": format_model_name(sv_model),
        "TTS_model": tts.__class__.__name__,
        "attacker": attacker_name,
        "TTS_query": total_tts_cnt,
        "time_cost": total_time_cost,
        "attack_strategy": "maximize_similarity",
        "total_samples_processed": len(all_results),
        "metadata": {
            "experiment_start_time": start_timestamp,
            "experiment_end_time": end_timestamp,
            "total_runtime_seconds": total_runtime,
            "total_runtime_formatted": format_runtime(total_runtime),
        },
        "results": all_results,
    }

    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    logger.info(f"Final consolidated results saved to {file_name}")

def format_data(data):
    """Format dataset for TextAttack compatibility"""
    _data = [(d['transcript'], 0) for d in data]
    return _data

def format_model_name(sv_model):
    """Format model name consistently"""
    return f"{sv_model.model_name}-{sv_model.dataset}-{sv_model.train_type}"

def format_runtime(seconds):
    """Format runtime consistently across the application"""
    if seconds is None:
        return None
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h:{minutes}m:{seconds%60:.2f}s"

def initialize_models(args):
    """Initialize speech verification and TTS models"""
    # Initialize speech verification model
    sv_model = SpeechVerificationModel(
        model_name=args.sv_model,
        dataset=args.dataset,
        train_type=args.train_type
    )
    
    # Initialize TTS model and set cache directory
    tts_name = args.tts
    tts_cls = TTS_MODELS_REGISTER[tts_name]
    tts = tts_cls(cache_dir=args.cache_dir)
    
    logger.info(f"TTS cache directory set to: {args.cache_dir}")
    logger.info(f"Results output directory: {args.output_dir}")
    
    return sv_model, tts

def load_dataset_and_setup(args):
    """Load target dataset which is filtered out from the original dataset below a threshold"""
    # Load dataset
    voxceleb_data = load_dataset(path=args.target_dataset_path)
    if args.max_examples:
        voxceleb_data = voxceleb_data[:args.max_examples]
        logger.info(f"Limited to {args.max_examples} examples for testing")
    
    return voxceleb_data

# def load_target_transcript_mapping() -> Dict[str, str]:
#     """Load target transcript mapping from CSV file"""
#     target_transcript_map = {}
#     csv_path = 'data/voxceleb_transcripts_test.csv'
#     if os.path.exists(csv_path):
#         with open(csv_path, 'r', encoding='utf-8') as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 # Get relative path from absolute audio_path
#                 abs_path = row['audio_path']
#                 # Find the part after .../test/ (should match target_audio)
#                 rel_idx = abs_path.find('/test/')
#                 if rel_idx != -1:
#                     rel_path = abs_path[rel_idx + len('/test/'):]
#                     target_transcript_map[rel_path] = row['full_transcript']
#     else:
#         logger.warning(f"Target transcript CSV not found: {csv_path}")
    # 
    # return target_transcript_map

def prepare_valid_entries(voxceleb_data, args):
    """Filter and prepare valid dataset entries"""
    valid_entries = []
    for idx, entry in enumerate(voxceleb_data):
        target_transcript = entry['target_transcript']
        ref_audio_rel = entry['ref_audio_rel']
        ref_audio_abs = os.path.join(args.ref_audio_root, ref_audio_rel)
        target_audio_rel = entry['target_audio_rel']
        target_audio_abs = os.path.join(args.ref_audio_root, target_audio_rel)
        
        # Defensive: check if reference audio exists
        if not os.path.exists(ref_audio_abs):
            logger.warning(f"Reference audio not found: {ref_audio_abs}, skipping entry {idx}")
            continue
        
        valid_entries.append({
            'original_idx': idx,
            'ref_audio_abs': ref_audio_abs,
            'target_audio_abs': target_audio_abs,
            'target_audio_rel': target_audio_rel,
            'target_transcript': target_transcript,
        })
    
    return valid_entries

def setup_attack_for_entry(entry, sv_model, tts, AttackerMethod, args):
    """Set up attack components for a single entry"""
    # Create model wrapper for this specific entry
    model_wrapper = SpeechVerificationPipelineWrapper(
        sv_model=sv_model,
        tts=tts,
        reference_audio_path=entry['ref_audio_abs'],
        target_audio_path=entry['target_audio_abs'],
        target_audio_rel_path=entry['target_audio_rel'],
        target_transcript=entry['target_transcript'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_seeding=args.use_seeding,
        seed_value=args.seed_value,
        use_cache=args.use_cache,
        cache_dir=args.cache_dir,
        original_fake_audios_path=args.original_fake_audios_path
    )
    
    # Build the attack recipe
    recipe = AttackerMethod.build(model_wrapper)
    transformation = recipe.transformation
    constraints = recipe.constraints
    search_method = recipe.search_method
    
    # Create custom goal function
    custom_goal = SimilarityMaximizationGoal(model_wrapper, threshold=args.similarity_threshold)
    
    # Manually construct the Attack object
    attack = Attack(
        goal_function=custom_goal,
        constraints=constraints,
        transformation=transformation,
        search_method=search_method,
    )
    
    # Create single-sample dataset
    sample_dataset = Dataset(dataset=[(entry['target_transcript'], 0)])
    attacker = Attacker(attack, sample_dataset)
    attacker.attack_args.num_examples = 1  # Attack this single example
    
    # --- Replace Universal Sentence Encoder (USE) with SBERT constraint ---
    use_idx = -1
    for idx_c, constraint in enumerate(attacker.attack.constraints):
        if isinstance(constraint, UniversalSentenceEncoder):
            use_idx = idx_c
            break
    
    window_size = 15
    compare_against_original = True
    if use_idx != -1:
        use_constraint = attacker.attack.constraints.pop(use_idx)
        window_size = use_constraint.window_size
        compare_against_original = use_constraint.compare_against_original
    
    SBERT_constraint = SBERT(
        model_name='sentence-transformers/all-MiniLM-L6-v2', 
        threshold=0.84,
        window_size=window_size,
        metric='angular',
        compare_against_original=compare_against_original,
        skip_text_shorter_than_window=False
    )
    attacker.attack.constraints.append(SBERT_constraint)
    # --- End SBERT replacement logic ---
    
    return model_wrapper, attacker, sample_dataset

def process_single_entry(entry, entry_idx, sv_model, tts, AttackerMethod, args, main_attacker):
    """Process a single dataset entry and return results"""
    logger.info(f"Processing entry {entry_idx + 1}: {entry['target_audio_rel']}")
    
    # Set up attack for this entry
    model_wrapper, attacker, sample_dataset = setup_attack_for_entry(
                                        entry,
                                        sv_model,
                                        tts,
                                        AttackerMethod,
                                        args)
    
    # Attack this single sample
    sample_results = attacker.attack_dataset()
    
    # Extract and format result for this sample
    attack_results = getattr(attacker.attack_log_manager, 'results', [])
    
    # Initialize main_attacker with the first attacker, then accumulate results
    if main_attacker[0] is None:
        main_attacker[0] = attacker
    else:
        # Add this attacker's results to the main attacker's results
        if attack_results:
            main_log_manager = getattr(main_attacker[0], 'attack_log_manager', None)
            if main_log_manager is not None and hasattr(main_log_manager, 'results'):
                # Directly extend the main attacker's results list
                main_log_manager.results.extend(attack_results)  # accumulate results
    
    # Format result
    item = None
    if attack_results and len(attack_results) > 0:
        result = attack_results[0]  # Should only be one result
        
        # Parse path: speaker_id/video_id/audio_filename
        try:
            path_parts = entry['target_audio_rel'].split('/')
            speaker_id = path_parts[0] if len(path_parts) > 0 else 'unknown'
            video_id = path_parts[1] if len(path_parts) > 1 else 'unknown'
            audio_filename = path_parts[2] if len(path_parts) > 2 else 'unknown'
        except Exception as e:
            logger.warning(f"Failed to extract audio details for entry {entry_idx}: {e}")
            speaker_id = 'unknown'
            video_id = 'unknown'
            audio_filename = 'unknown'
        
        original_score = result.original_result.score
        perturbed_score = result.perturbed_result.score
        
        item = {
            "speaker_id": speaker_id,
            "video_id": video_id,
            "audio_filename": audio_filename,
            "original_text": result.original_text(),
            "original_similarity": original_score,
            "perturbed_text": result.perturbed_text(),
            "perturbed_similarity": perturbed_score,
            "similarity_improvement": perturbed_score - original_score,
            "target_audio_path": entry['target_audio_rel'],
            "reference_audio_path": entry['ref_audio_abs'],
        }
    else:
        logger.warning(f"No attack results for entry {entry_idx}: {entry['target_audio_rel']}")
    
    # Get TTS count for statistics
    tts_count = model_wrapper.tts_cnt
    
    # Clean up this entry's objects
    del model_wrapper, attacker, sample_dataset
    gc.collect()
    
    return item, tts_count

def process_batch(
    batch_entries, 
    batch_idx, 
    total_batches, 
    sv_model, 
    tts, 
    AttackerMethod, 
    args, 
    main_attacker
):
    """Process a batch of entries"""
    # Record batch start time
    batch_start_time = datetime.datetime.now()
    
    logger.info(f"Processing batch {batch_idx + 1}/{total_batches} (samples {len(batch_entries)})")
    logger.info(f"Batch start time: {batch_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Log which speakers are in this batch
    batch_speakers = set()
    for entry in batch_entries:
        speaker_id = entry['target_audio_rel'].split('/')[0] if '/' in entry['target_audio_rel'] else 'unknown'
        batch_speakers.add(speaker_id)
    logger.info(f"Batch {batch_idx + 1} contains speakers: {sorted(batch_speakers)}")
    
    # Process each entry individually
    batch_items = []
    batch_tts_count = 0
    
    for entry_idx, entry in enumerate(batch_entries):
        item, tts_count = process_single_entry(entry, entry_idx, sv_model, tts, AttackerMethod, args, main_attacker)
        if item:
            batch_items.append(item)
        batch_tts_count += tts_count
    
    # Calculate and log batch completion time
    batch_end_time = datetime.datetime.now()
    batch_runtime = (batch_end_time - batch_start_time).total_seconds()
    batch_runtime_formatted = format_runtime(batch_runtime)
    
    logger.info(f"Batch {batch_idx + 1}/{total_batches} completed. "
               f"Runtime: {batch_runtime_formatted}. "
               f"Processed {len(batch_items)} samples.")
    
    return batch_items, batch_tts_count, batch_start_time

def run_attacker_experiment(attacker_name, AttackerMethod, valid_entries, sv_model, tts, args):
    """Run complete experiment for one attacker method"""
    
    # Set up output file
    model_name_str = format_model_name(sv_model).replace('-', '_')
    file_name = os.path.join(args.output_dir, f"{attacker_name}_vox1-O_{model_name_str}_{args.seed_value}.json")
    do_skip = args.skip == "True"
    
    if os.path.exists(file_name) and do_skip:
        logger.info(f"Skipping existing result: {file_name}")
        return
    
    # Record attacker start time
    attacker_start_time = datetime.datetime.now()
    attacker_start_timestamp = attacker_start_time.isoformat()
    
    BATCH_SIZE = args.batch_size
    
    logger.info(f"Running {attacker_name} attack on all pairs with batch size {BATCH_SIZE}")
    logger.info(f"Attacker start time: {attacker_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Log speaker distribution in valid entries
    speaker_counts = {}
    for entry in valid_entries:
        speaker_id = entry['target_audio_rel'].split('/')[0] if '/' in entry['target_audio_rel'] else 'unknown'
        speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
    
    logger.info(f"Speaker distribution: {speaker_counts}")
    
    # Calculate number of batches
    total_batches = math.ceil(len(valid_entries) / BATCH_SIZE)
    logger.info(f"Processing in {total_batches} batches of {BATCH_SIZE} samples each")
    
    # Track overall statistics
    all_consolidated_results = []
    total_tts_count = 0
    
    # Create a main attacker object to collect all results for log summary
    main_attacker = [None]  # Use list to allow modification in nested functions
    
    # Process in batches
    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(valid_entries))
        batch_entries = valid_entries[start_idx:end_idx]
        
        batch_items, batch_tts_count, batch_start_time = process_batch(
            batch_entries, batch_idx, total_batches, sv_model, tts, AttackerMethod, args, main_attacker
        )
        
        # Update overall statistics
        total_tts_count += batch_tts_count
        all_consolidated_results.extend(batch_items)
        
        # Save partial results for this batch
        save_partial_results(
            batch_items, attacker_name, sv_model, tts, 
            file_name, batch_idx + 1, total_batches, batch_start_time
        )
        
        logger.info(f"Total results so far: {len(all_consolidated_results)}")
    
    # Save final consolidated results
    if all_consolidated_results:
        # Calculate attacker end time and total runtime
        attacker_end_time = datetime.datetime.now()
        attacker_end_timestamp = attacker_end_time.isoformat()
        attacker_runtime_seconds = (attacker_end_time - attacker_start_time).total_seconds()
        
        # Calculate final time cost
        final_time_cost = format_runtime(attacker_runtime_seconds)
        
        # Generate log summary from the main attacker
        log_summary = None
        if main_attacker[0] is not None:
            log_summary = get_attack_log_summary(main_attacker[0])
        
        save_final_consolidated_results(
            all_consolidated_results, attacker_name, sv_model, tts, 
            file_name, total_tts_count, final_time_cost,
            start_timestamp=attacker_start_timestamp,
            end_timestamp=attacker_end_timestamp,
            total_runtime=attacker_runtime_seconds,
            log_summary=log_summary
        )
        
        logger.info(f"Attack {attacker_name} completed. Runtime: {final_time_cost}")
        logger.info(f"Final results saved to {file_name}")
    else:
        logger.warning(f"No results to save for {attacker_name}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Speech Verification Attack Script')
    parser.add_argument('--sv_model', type=str, default='b6', 
                       help='ReDimNet model to attack (M, S, b0-b6)')
    parser.add_argument('--dataset', type=str, default='vox2',
                       help='ReDimNet dataset (vb2, vox2, vb2+vox2+cnc)')
    parser.add_argument('--train_type', type=str, default='ft_lm',
                       help='ReDimNet training type (ptn, ft_lm, ft_mix)')
    parser.add_argument('--tts', type=str, default='F5TTS', 
                       help='TTS model to use')
    parser.add_argument('--skip', type=str, default='True', 
                       help='Skip existing attack results')
    parser.add_argument("--target_dataset_path", type=str, default='sv_simple_attack_results_05072025_16khz/all-filtered-cases_0.63.json')
    parser.add_argument('--output_dir', type=str, default='real_attack_results',
                       help='Output directory for anlysis json results')
    parser.add_argument('--max_examples', type=int, default=None,
                       help='Maximum number of examples to attack')
    parser.add_argument('--ref_audio_root', type=str, default='/media/volume/AudioUnlearnData1/emnlp_audio_adversarial/data/test/',
                       help='Root directory for reference audio files (absolute path)')
    parser.add_argument('--cache_dir', type=str, default='./cache/fake_generated_audios',
                       help='Directory to save generated fake audio during attack')
    parser.add_argument('--original_fake_audios_path', type=str, default='./cache/original-fake-audios-05072025_16khz',
                       help='Path to original fake audio files generated before')
    parser.add_argument('--use_seeding', action='store_true', default=True,
                       help='Use deterministic seeding (default: True)')
    parser.add_argument('--no_seeding', dest='use_seeding', action='store_false',
                       help='Disable deterministic seeding')
    parser.add_argument('--seed_value', type=int, default=16,
                       help='Seed value for deterministic seeding')
    parser.add_argument('--use_cache', action='store_true', default=True,
                       help='Use caching for generated audio (default: True)')
    parser.add_argument('--no_cache', dest='use_cache', action='store_false',
                       help='Disable caching - always generate fresh audio')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='Batch size for processing samples (default: 10)')
    parser.add_argument('--similarity_threshold', type=float, default=0.7,
                       help='Similarity threshold for attack (default: 0.68)')

    return parser.parse_args()

if __name__ == "__main__":
    # Record experiment start time
    experiment_start_time = datetime.datetime.now()
    
    # Parse arguments first
    args = parse_arguments()

    attacker_name = list(ATTACKER_MAP.keys())[0]
    args.cache_dir = args.cache_dir + f'/{attacker_name}_seed_{args.seed_value}'
    # Setup all directories and logging
    log_dir = setup_directories(args)
    setup_logging(log_dir, args.tts, attacker_name, args.seed_value)
    
    logger.info(f"=== Speech Verification Attack Experiment Started ===")
    logger.info(f"Start time: {experiment_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Cache directory: {args.cache_dir}")
    
    # Initialize models
    sv_model, tts = initialize_models(args)
    
    # Load dataset
    voxceleb_data = load_dataset_and_setup(args)
    
    # Load target transcript mapping
    # target_transcript_map = load_target_transcript_mapping()
    
    # Prepare valid entries
    valid_entries = prepare_valid_entries(voxceleb_data, args)
    
    if not valid_entries:
        logger.error("No valid samples found to attack")
        exit(1)
    
    logger.info(f"Starting speech verification attacks (per transcript/ref_audio pair)...")
    logger.info(f"SV Model: {args.sv_model}-{args.dataset}-{args.train_type}")
    logger.info(f"TTS Model: {args.tts}")
    logger.info(f"Cache Dir: {args.cache_dir}")
    logger.info(f"Output Dir: {args.output_dir}")
    logger.info(f"Reference Audio Root: {args.ref_audio_root}")
    logger.info(f"Total pairs: {len(voxceleb_data)}")
    logger.info(f"Valid pairs: {len(valid_entries)}")

    # Run experiments for each attacker method
    for attacker_name, AttackerMethod in ATTACKER_MAP.items():
        run_attacker_experiment(attacker_name, AttackerMethod, valid_entries, sv_model, tts, args)
    
    # Calculate total experiment runtime
    experiment_end_time = datetime.datetime.now()
    total_experiment_runtime = (experiment_end_time - experiment_start_time).total_seconds()
    
    logger.info("=== All speech verification attacks completed! ===")
    logger.info(f"Experiment end time: {experiment_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total experiment runtime: {format_runtime(total_experiment_runtime)}")
    logger.info(f"Total experiment runtime: {total_experiment_runtime:.2f} seconds")