#!/usr/bin/env python3
"""
VoxCeleb Speech Verification Attack Script using SSL Models
Performs voice cloning attacks on SSL-based speech verification systems using VoxCeleb1-O protocol

python original_simple_attack_f5tts_16khz_ssl.py --ssl_config "models/tests/simclr/simclr_e-ecapa/ssps_kmeans_25k_uni-1/config.yml" --ssl_checkpoint "model_avg.pt" --max_trials 3 --eer_method interpolation --tts "F5TTS"


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
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm
import librosa
import soundfile as sf
import shutil
from loguru import logger
import warnings
import datetime
import tempfile
import sys

# Add notebooks path for SSL model utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'notebooks'))

# Import SSL model utilities
from notebooks.notebooks_utils import load_models, Model
from sslsv.evaluations.CosineSVEvaluation import CosineSVEvaluation, CosineSVEvaluationTaskConfig

# TTS utilities 
from tts_utils import TTS_MODELS_REGISTER, TTSGenerator, CoquiTTS, F5TTSGenerator, OpenAITTS, KokoroTTS

# Set environment variable for CUDA deterministic behavior
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Set deterministic behavior for PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

def setup_directories(output_dir: str, cache_dir: str = "cache"):
    """Create all necessary directories for the experiment"""
    # Create output directory and its subdirectories
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    return log_dir

def setup_logging(log_dir: str):
    """Configure loguru with proper log directory"""
    logger.remove()  # Remove default handler
    logger.add(
        os.path.join(log_dir, "original_simple_attack_F5TTS_SSL.log"),
        format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
        level="INFO",
        rotation="10 MB"
    )
    logger.add(
        lambda msg: print(msg, end=""),  # Console handler
        format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}\n",
        level="INFO"
    )

class SSLSpeechVerificationModel:
    """Wrapper for SSL-based speech verification models"""
    
    def __init__(self, config_path: str, checkpoint_name: str = "model_latest.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config_path = config_path
        self.checkpoint_name = checkpoint_name
        self.model_entry: Optional[Model] = None
        self.load_model()
        
    def load_model(self):
        """Load SSL model using the standard SSL loading pattern"""
        try:
            logger.info(f"Loading SSL model from config: {self.config_path}")
            logger.info(f"Using checkpoint: {self.checkpoint_name}")
            
            models = load_models([self.config_path], checkpoint_name=self.checkpoint_name)
            
            if not models:
                raise ValueError("No models loaded")
            
            # Get the first (and only) model
            self.model_entry = list(models.values())[0]
            
            # Patch model configs to avoid memory issues
            self.model_entry.config.dataset.num_workers = 0
            self.model_entry.config.dataset.pin_memory = False
            
            # Set the correct base_path for the dataset
            if not hasattr(self.model_entry.config.dataset, 'base_path') or self.model_entry.config.dataset.base_path is None:
                self.model_entry.config.dataset.base_path = Path('/media/volume/AudioUnlearnData1/sslsv/data/')
            else:
                self.model_entry.config.dataset.base_path = Path('/media/volume/AudioUnlearnData1/sslsv/data/')
            
            logger.info("SSL model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SSL model: {e}")
            raise
    
    def extract_embedding(self, audio: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract embedding from audio using the SSL model"""
        try:
            if self.model_entry is None or self.model_entry.model is None:
                logger.error("Model is not loaded")
                return None
                
            with torch.no_grad():
                # Add batch dimension if needed
                if audio.dim() == 1:
                    audio_batch = audio.unsqueeze(0).to(self.device)
                else:
                    audio_batch = audio.to(self.device)
                
                # Extract embedding using the SSL model
                embedding = self.model_entry.model(audio_batch)
                
                # Normalize embedding (following SSL evaluation pattern)
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
                
                return embedding.squeeze(0).cpu()
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            return None
    
    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between two embeddings"""
        return (torch.dot(emb1, emb2).item() + 1) / 2  # Convert from [-1, 1] to [0, 1]

class VoxCelebSSLAttacker:
    """
    VoxCeleb speech verification attacker using voice cloning against SSL models
    """
    
    def __init__(
        self, 
        sv_model: SSLSpeechVerificationModel, 
        tts: TTSGenerator, 
        voxceleb_root: str = "/media/volume/AudioUnlearnData1/sslsv/data/", 
        protocol_file: str = "voxceleb1_test_O",
        target_transcript_file: str = "voxceleb_transcripts_test.csv", 
        use_cache: bool = True, 
        cache_dir: str = "cache",
        use_seeding: bool = True,
        seed_value: int = 75,
        eer_method: str = "discrete"
    ):
        self.sv_model = sv_model
        self.tts = tts
        self.voxceleb_root = voxceleb_root
        self.protocol_file = protocol_file
        self.target_transcript_file = target_transcript_file   
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.use_seeding = use_seeding 
        self.seed_value = seed_value
        self.eer_method = eer_method
        # Load transcripts from CSV
        self.target_transcript_map = self._load_all_transcripts()
        
        # Compute optimal verification threshold (EER) for this SSL model
        # Use "interpolation" for more precision or "discrete" for SSL framework consistency
        # self.verification_threshold = self._compute_eer_threshold(method=self.eer_method)
        # discrete: 0.29427021741867065 
        # interpolation: 0.2942563326819349
        self.verification_threshold  = (0.294 + 1) / 2

        # Statistics
        self.total_trials = 0
        self.successful_attacks = 0
        self.failed_attacks = 0
        self.processing_errors = 0
        
    def _load_all_transcripts(self) -> Dict[str, str]:
        """Load transcripts from CSV file using SSL framework pattern"""
        target_transcript_map = {}
        
        # Get the base path from SSL model config (following SSL framework pattern)
        base_path = self.sv_model.model_entry.config.dataset.base_path # PosixPath('/media/volume/AudioUnlearnData1/sslsv/data')
        
        transcript_path = base_path / self.target_transcript_file

        try:
            logger.info(f"Loading transcripts from: {transcript_path}")
            df = pd.read_csv(transcript_path)
            for _, row in df.iterrows():
                # Use audio_path as key (relative path from voxceleb root)
                audio_path = str(row['audio_path'])
                transcript = str(row['full_transcript'])
                
                # Key 3: Extract relative path from full path and normalize
                if '/test/' in audio_path:
                    relative_path = audio_path.split('/test/')[-1]

                # Key 4: Add voxceleb1 prefix to match protocol format
                if not relative_path.startswith('voxceleb1/'):
                    voxceleb1_path = f"voxceleb1/{relative_path}"
                    target_transcript_map[voxceleb1_path] = transcript

            logger.info(f"Loaded {len(df)} transcripts from {transcript_path}")
            logger.info(f"Created {len(target_transcript_map)} transcript lookup keys for robust matching")

            # Debug: Show some example keys for the first entry
            if target_transcript_map:
                first_keys = [k for k in list(target_transcript_map.keys())[:5]]
                logger.debug(f"Example transcript keys: {first_keys}")
            
            return target_transcript_map
        except Exception as e:
            logger.error(f"Failed to load transcripts from {transcript_path}: {e}")
            return {}
    
    def _compute_eer_threshold(self, method: str = "discrete") -> float:
        """Compute the EER (Equal Error Rate) threshold for this SSL model
        
        Args:
            method: "discrete" (SSL framework style) or "interpolation" (more precise)
        """
        try:
            logger.info(f"Computing EER threshold for SSL model using {method} method...")
            
            # Create a minimal evaluation to get scores
            eval_task_config = CosineSVEvaluationTaskConfig(
                __type__="sv_cosine",
                frame_length=None,
                num_frames=1,
                trials=['voxceleb1_test_O'],  # Use same trial as attacks
                metrics=['eer']
            )
            
            # Create evaluation instance 
            eval_instance = CosineSVEvaluation(
                model=self.sv_model.model_entry.model,
                config=self.sv_model.model_entry.config,
                task_config=eval_task_config,
                device=self.sv_model.device,
                verbose=False,
                validation=False
            )
            
            # Run evaluation to get scores
            metrics = eval_instance.evaluate()
            scores = eval_instance.scores
            targets = eval_instance.targets
            
            if method == "interpolation":
                # User's method: More mathematically precise using interpolation
                from sklearn.metrics import roc_curve
                from scipy.optimize import brentq
                from scipy.interpolate import interp1d
                
                fpr, tpr, thresholds = roc_curve(targets, scores, pos_label=1)
                eer_rate = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
                eer_threshold = interp1d(fpr, thresholds)(eer_rate)
                
                # Ensure scalar output
                if hasattr(eer_threshold, 'item'):
                    eer_threshold = float(eer_threshold.item())
                elif isinstance(eer_threshold, np.ndarray):
                    eer_threshold = float(eer_threshold.squeeze())
                else:
                    eer_threshold = float(eer_threshold)
                    
                logger.info(f"Computed EER threshold (interpolation): {eer_threshold:.4f} (EER rate: {eer_rate:.4f})")
                
            else:
                # Discrete method: SSL framework style
                scores_array = np.array(scores)
                targets_array = np.array(targets)
                
                nb_target_scores = len(scores_array[targets_array == 1])
                nb_nontarget_scores = len(scores_array[targets_array == 0])
                
                sorted_idx = np.argsort(scores_array)
                sum_fn = np.cumsum(targets_array[sorted_idx])
                sum_fp = np.cumsum(np.where(targets_array[sorted_idx] == 0, 1, 0))
                
                fnrs = np.empty(len(scores_array) + 1)
                fnrs[0] = 0
                fnrs[1:] = sum_fn / nb_target_scores
                
                fprs = np.empty(len(scores_array) + 1)
                fprs[0] = 1
                fprs[1:] = (nb_nontarget_scores - sum_fp) / nb_nontarget_scores
                
                sorted_scores = scores_array[sorted_idx]
                
                # Find EER threshold
                idx = np.nanargmin(np.abs(fnrs - fprs))
                eer_rate = max(fprs[idx], fnrs[idx])
                eer_threshold = sorted_scores[idx] if idx < len(sorted_scores) else sorted_scores[-1]
                
                logger.info(f"Computed EER threshold (discrete): {eer_threshold:.4f} (EER rate: {eer_rate:.4f})")
            
            return float(eer_threshold)
            
        except Exception as e:
            logger.warning(f"Failed to compute EER threshold: {e}")
            logger.warning("Using default threshold: 0.68")
            return 0.68
        
    def load_protocol(self) -> List[Tuple[int, str, str]]:
        """Load VoxCeleb1-O test protocol using SSL framework pattern"""
        trials = []
        
        # Get the base path from SSL model config (following SSL framework pattern)
        base_path = self.sv_model.model_entry.config.dataset.base_path # PosixPath('/media/volume/AudioUnlearnData1/sslsv/data')

        protocol_path = base_path / self.protocol_file

        if protocol_path is None:
            logger.error("Please check that the protocol file exists or provide the correct path.")
            return []
        
        try:
            logger.info(f"Loading protocol from: {protocol_path}")
            with open(protocol_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        label = int(parts[0])
                        ref_audio = parts[1]  # We'll ignore this as specified
                        target_audio = parts[2]  # We'll use this for voice cloning
                        trials.append((label, ref_audio, target_audio))
            logger.info(f"Loaded {len(trials)} trials from protocol file")
            return trials
        except Exception as e:
            logger.error(f"Failed to load protocol from {protocol_path}: {e}")
            return []
    
    def load_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """Load audio file from VoxCeleb dataset"""
        full_path = os.path.join(self.voxceleb_root, audio_path)
        try:
            audio, sr = librosa.load(full_path, sr=16000)
            return audio
        except Exception as e:
            logger.error(f"Failed to load audio {full_path}: {e}")
            return None
    
    def extract_transcript(self, audio_path: str) -> Optional[str]:
        """Get transcript from pre-loaded transcripts"""
        try:
            # Try exact match first
            if audio_path in self.target_transcript_map:
                return self.target_transcript_map[audio_path]

            logger.warning(f"No transcript found for {audio_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to get transcript for {audio_path}: {e}")
            return None
    
    def get_audio_path(self, audio_path: str) -> str:
        """Get full path to audio file for voice cloning, avoiding duplicate voxceleb1/ prefix"""
        if os.path.isabs(audio_path):
            return audio_path
        # Remove leading voxceleb1/ if present
        if audio_path.startswith("voxceleb1/"):
            audio_path = audio_path[len("voxceleb1/"):]
        return os.path.join(self.voxceleb_root, audio_path)
    
    def _set_deterministic_seed(self):
        """Set deterministic seed for all random number generators"""
        torch.manual_seed(self.seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed_value)
            torch.cuda.manual_seed_all(self.seed_value)
        
        import random
        import numpy as np
        random.seed(self.seed_value)
        np.random.seed(self.seed_value)

    def generate_fake_audio(
        self, 
        target_transcript: str, 
        target_audio_abs: str, 
    ) -> Optional[str]:
        """Generate fake audio using voice cloning TTS with seeding and cache control"""
        try:
            # Conditionally set deterministic seed
            if self.use_seeding:
                self._set_deterministic_seed()
            
            tts_method = self.tts.__class__.__name__
            
            if self.use_cache:
                parts = target_audio_abs.split('/')
                if len(parts) >= 3:
                    speaker_id, video_id, audio_filename = parts[-3], parts[-2], parts[-1]

                # Create cache path with seeding mode
                seeding_mode = "seeded" if self.use_seeding else "unseeded"
                # cache_root = os.path.join(self.cache_dir, f"ssl-fake-audios-05072025_16khz/{tts_method}/{seeding_mode}")
                cache_root = os.path.join(self.cache_dir, f"original-fake-audios-05072025_16khz/{tts_method}/{seeding_mode}")
                if self.use_seeding:
                    # Include seed in filename when seeding is used
                    name, ext = os.path.splitext(audio_filename)
                    seeded_filename = f"{name}_seed{self.seed_value}{ext}"
                    cache_audio_path = os.path.join(cache_root, speaker_id, video_id, seeded_filename)
                else:
                    # Use original filename when seeding is not used
                    cache_audio_path = os.path.join(cache_root, speaker_id, video_id, audio_filename)
                
                cache_dir_path = os.path.dirname(cache_audio_path)
                os.makedirs(cache_dir_path, exist_ok=True)
                
                # Use cache if available
                if os.path.exists(cache_audio_path):
                    logger.info(f"Using cached audio: {cache_audio_path}")
                    return cache_audio_path
            else:
                # No caching - use temporary file
                temp_dir = tempfile.mkdtemp(prefix="nocache_ssl_attack_")
                audio_filename = os.path.basename(target_audio_abs)
                if self.use_seeding:
                    name, ext = os.path.splitext(audio_filename)
                    seeded_filename = f"{name}_seed{self.seed_value}{ext}"
                    cache_audio_path = os.path.join(temp_dir, seeded_filename)
                else:
                    cache_audio_path = os.path.join(temp_dir, audio_filename)
                
                logger.info(f"Generating fresh audio (no cache): {os.path.basename(cache_audio_path)}")
            
            # Generate fake audio based on TTS model type
            if isinstance(self.tts, CoquiTTS):
                # Use voice cloning with the new cleaner interface
                self.tts.set_speaker_wav(target_audio_abs)

                if self.tts.speaker_wav != target_audio_abs:
                    logger.warning(f"Speaker wav not set correctly: {self.tts.speaker_wav} vs {target_audio_abs}")

                try:
                    # Use the standard generate_audio method (handles long text automatically)
                    fake_audio_list = self.tts.generate_audio([target_transcript], [cache_audio_path])
                    
                    if fake_audio_list and os.path.exists(fake_audio_list[0]):
                        logger.info(f"Generated audio saved: {cache_audio_path}")
                        return cache_audio_path
                        
                except Exception as e:
                    logger.error(f"CoquiTTS generation failed: {e}")
                    
            elif isinstance(self.tts, F5TTSGenerator):
                # Use voice cloning with the new cleaner interface
                self.tts.set_reference_audio(target_audio_abs, target_transcript)
                if self.tts.ref_audio != target_audio_abs:
                    logger.warning(f"Reference audio not set correctly: {self.tts.ref_audio} vs {target_audio_abs}")
                
                try:
                    self.tts.single_infer(target_transcript, cache_audio_path)
                    
                    if os.path.exists(cache_audio_path):
                        logger.info(f"Generated audio saved: {cache_audio_path}")
                        return cache_audio_path
                        
                except Exception as e:
                    logger.error(f"F5TTS generation failed: {e}")
                    
            elif isinstance(self.tts, KokoroTTS):
                # Note: KokoroTTS doesn't support voice cloning from arbitrary audio files
                # It uses predefined voices only. For voice cloning attacks, use CoquiTTS or F5TTS instead.
                try:
                    fake_audio_list = self.tts.generate_audio([target_transcript], [cache_audio_path])
                    
                    if fake_audio_list and os.path.exists(fake_audio_list[0]):
                        logger.info(f"Generated audio saved (using predefined voice '{getattr(self.tts, 'voice', 'default')}'): {cache_audio_path}")
                        return cache_audio_path
                        
                except Exception as e:
                    logger.error(f"KokoroTTS generation failed: {e}")
            
            elif isinstance(self.tts, OpenAITTS):
                # Note: OpenAI TTS doesn't support voice cloning from arbitrary audio files
                # It uses predefined voices only. For voice cloning attacks, use CoquiTTS or F5TTS instead.
                try:
                    fake_audio_list = self.tts.generate_audio([target_transcript], [cache_audio_path])
                    
                    if fake_audio_list and os.path.exists(fake_audio_list[0]):
                        logger.info(f"Generated audio saved (using predefined voice '{getattr(self.tts, 'voice', 'default')}'): {cache_audio_path}")
                        return cache_audio_path
                        
                except Exception as e:
                    logger.error(f"OpenAI TTS generation failed: {e}")
            
            else:
                # For other TTS models, use standard generation
                fake_audio_list = self.tts([target_transcript])
                if fake_audio_list and fake_audio_list[0]:
                    temp_audio_path = fake_audio_list[0]
                    if os.path.exists(temp_audio_path):
                        shutil.move(temp_audio_path, cache_audio_path)
                        logger.info(f"Generated audio saved: {cache_audio_path}")
                        return cache_audio_path
            
            return None
                
        except Exception as e:
            logger.error(f"Failed to generate fake audio: {e}")
            return None
    
    def compute_similarity(self, audio1_path: str, audio2: np.ndarray) -> Optional[float]:
        """Compute similarity between reference audio and generated audio"""
        try:
            # Load reference audio
            audio1 = self.load_audio(audio1_path)
            if audio1 is None:
                return None
            
            # Extract embeddings using SSL model
            audio1_tensor = torch.tensor(audio1, dtype=torch.float32)
            audio2_tensor = torch.tensor(audio2, dtype=torch.float32)
            
            emb1 = self.sv_model.extract_embedding(audio1_tensor)
            emb2 = self.sv_model.extract_embedding(audio2_tensor)
            
            if emb1 is None or emb2 is None:
                return None
            
            # Compute cosine similarity
            similarity = self.sv_model.compute_similarity(emb1, emb2) 
            return similarity
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return None
    
    def attack_single_trial(
        self, 
        label: int, 
        ref_audio_rel: str, 
        target_audio_rel: str, 
        ) -> Dict:
        """Attack a single trial from the protocol"""
        result = {
            'label': label,
            'ref_audio_rel': ref_audio_rel,
            'target_audio_rel': target_audio_rel,
            'success': False,
            'original_similarity': None,
            'fake_similarity': None,
            'target_transcript': None,
            'error': None,
            'speaker_id': None,
            'video_id': None,
            'audio_filename': None
        }
        
        try:
            # Parse path: speaker_id/video_id/audio_filename
            try:
                path_parts = target_audio_rel.split('/')
                result['speaker_id'] = path_parts[0] if len(path_parts) > 0 else 'unknown'
                result['video_id'] = path_parts[1] if len(path_parts) > 1 else 'unknown'
                result['audio_filename'] = path_parts[2] if len(path_parts) > 2 else 'unknown'
            except Exception as e:
                logger.warning(f"Failed to extract audio details for {target_audio_rel}: {e}")
            
            # Step 1: Extract transcript from target audio
            target_transcript = self.extract_transcript(target_audio_rel)

            if not target_transcript:
                result['error'] = 'Failed to extract transcript'
                return result
            
            result['target_transcript'] = target_transcript
            
            try:
                # Step 2: Compute original similarity between ref and target
                print(f"Loading target audio: {target_audio_rel}")
                target_audio_data = self.load_audio(target_audio_rel)
                if target_audio_data is None:
                    result['error'] = 'Failed to load target audio'
                    return result
                
                original_similarity = self.compute_similarity(ref_audio_rel, target_audio_data)
                if original_similarity is None:
                    result['error'] = 'Failed to compute original similarity'
                    return result
                
                result['original_similarity'] = original_similarity
                
                # Step 3: Generate fake audio using voice cloning with seeding
                target_audio_abs = os.path.join(self.voxceleb_root, target_audio_rel)
                fake_audio_path = self.generate_fake_audio(target_transcript, target_audio_abs) 
                if not fake_audio_path:
                    result['error'] = 'Failed to generate fake audio'
                    return result
                
                # Step 4: Compute similarity between ref and fake audio
                fake_audio_data, _ = librosa.load(fake_audio_path, sr=16000)
                fake_similarity = self.compute_similarity(ref_audio_rel, fake_audio_data)
                if fake_similarity is None:
                    result['error'] = 'Failed to compute fake similarity'
                    return result
                
                result['fake_similarity'] = fake_similarity
                
                # Step 5: Determine if attack was successful
                # For same speaker pairs (label=1), we want high similarity
                # For different speaker pairs (label=0), we want to fool the system into thinking they're the same
                threshold = getattr(self, 'verification_threshold', 0.68)  # Use model-specific threshold if available
                

                if label == 1:
                    # Same speaker: attack succeeds if fake similarity remains high
                    result['success'] = fake_similarity >= threshold
                else:
                    # Different speakers: attack succeeds if fake similarity becomes high
                    result['success'] = fake_similarity >= threshold and fake_similarity > original_similarity
                    
            finally:
                pass
                    
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error in attack_single_trial: {e}")
        
        return result
    
    def save_partial_results(self, batch_results: List[Dict], output_file: str, batch_num: int, total_batches: int, batch_start_time: Optional[datetime.datetime] = None):
        """Save partial results after processing a batch"""
        logger.info(f"Saving partial results - batch {batch_num}/{total_batches}")
        
        # Create partial filename
        partial_file_name = output_file.replace('.json', f'_partial_batch_{batch_num}.json')
        
        # Calculate batch timing if provided
        batch_metadata: Dict[str, Union[int, str, float, None]] = {
            "batch_number": batch_num,
            "total_batches": total_batches,
            "results_in_batch": len(batch_results)
        }
        
        if batch_start_time:
            batch_end_time = datetime.datetime.now()
            batch_runtime = (batch_end_time - batch_start_time).total_seconds()
            batch_metadata["batch_start_time"] = batch_start_time.isoformat()
            batch_metadata["batch_end_time"] = batch_end_time.isoformat()
            batch_metadata["batch_runtime_seconds"] = batch_runtime
            batch_metadata["batch_runtime_formatted"] = self._format_runtime(batch_runtime)
        
        data = {
            "ssl_model_config": self.sv_model.config_path,
            "ssl_model_checkpoint": self.sv_model.checkpoint_name,
            "tts_model": self.tts.__class__.__name__,
            "attack_strategy": "voice_cloning_ssl",
            "verification_threshold": self.verification_threshold,
            "eer_method": self.eer_method,
            "batch_info": batch_metadata,
            "results": batch_results,
        }

        with open(partial_file_name, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Partial results saved to {partial_file_name}")

    def _format_runtime(self, seconds: Optional[float]) -> Optional[str]:
        """Format runtime consistently"""
        if seconds is None:
            return None
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h:{minutes}m:{seconds%60:.2f}s"

    def run_attack(self, max_trials: Optional[int] = None, use_seeding: bool = True, batch_size: int = 50, output_file: Optional[str] = None) -> Dict:
        """Run attack on VoxCeleb1-O protocol with batch processing"""
        logger.info("Loading VoxCeleb1-O protocol...")
        trials = self.load_protocol()
        
        if not trials:
            logger.error("No trials loaded from protocol file")
            return {}
        
        if max_trials:
            trials = trials[:max_trials]
            logger.info(f"Limited to {max_trials} trials for testing")
        
        logger.info(f"Total trials: {len(trials)}")
        logger.info(f"Using seeding: {use_seeding}, Using cache: {self.use_cache}")
        logger.info(f"Batch size: {batch_size}")
        
        results = []
        start_time = time.time()
        experiment_start_time = datetime.datetime.now()
        
        # Calculate number of batches
        total_batches = math.ceil(len(trials) / batch_size)
        logger.info(f"Processing in {total_batches} batches")
        
        # Process in batches
        for batch_idx in range(total_batches):
            batch_start_time = datetime.datetime.now()
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(trials))
            batch_trials = trials[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} (trials {start_idx+1}-{end_idx})")
            
            batch_results = []
            for i, (label, ref_audio_rel, target_audio_rel) in enumerate(tqdm(batch_trials, desc=f"Batch {batch_idx+1}")):
                self.total_trials += 1
                
                result = self.attack_single_trial(label, ref_audio_rel, target_audio_rel)
                batch_results.append(result)
                results.append(result)
                
                if result['error']:
                    self.processing_errors += 1
                elif result['success']:
                    self.successful_attacks += 1
                else:
                    self.failed_attacks += 1
                
                # Log progress within batch
                if (i + 1) % 10 == 0:
                    success_rate = self.successful_attacks / max(1, self.total_trials - self.processing_errors) * 100
                    logger.info(f"Batch {batch_idx+1} progress: {i+1}/{len(batch_trials)}, Overall success rate: {success_rate:.1f}%")
            
            # Save partial results for this batch
            if output_file:
                self.save_partial_results(batch_results, output_file, batch_idx + 1, total_batches, batch_start_time)
            
            # Log batch completion
            batch_end_time = datetime.datetime.now()
            batch_runtime = (batch_end_time - batch_start_time).total_seconds()
            logger.info(f"Batch {batch_idx + 1}/{total_batches} completed. "
                       f"Runtime: {self._format_runtime(batch_runtime)}. "
                       f"Processed {len(batch_results)} samples.")
        
        # Compute final statistics
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Separate results by label
        same_speaker_results = [r for r in results if r['label'] == 1 and not r['error']]
        diff_speaker_results = [r for r in results if r['label'] == 0 and not r['error']]
        
        # Compute attack success rates
        same_speaker_success = sum(1 for r in same_speaker_results if r['success'])
        diff_speaker_success = sum(1 for r in diff_speaker_results if r['success'])
        
        attack_summary = {
            'total_trials': self.total_trials,
            'processing_errors': self.processing_errors,
            'successful_trials': self.total_trials - self.processing_errors,
            'same_speaker_trials': len(same_speaker_results),
            'diff_speaker_trials': len(diff_speaker_results),
            'same_speaker_attacks_success': same_speaker_success,
            'diff_speaker_attacks_success': diff_speaker_success,
            'same_speaker_success_rate': same_speaker_success / max(1, len(same_speaker_results)) * 100,
            'diff_speaker_success_rate': diff_speaker_success / max(1, len(diff_speaker_results)) * 100,
            'overall_success_rate': (same_speaker_success + diff_speaker_success) / max(1, len(same_speaker_results) + len(diff_speaker_results)) * 100,
            'processing_time': processing_time,
            'processing_time_formatted': self._format_runtime(processing_time),
            'experiment_start_time': experiment_start_time.isoformat(),
            'experiment_end_time': datetime.datetime.now().isoformat(),
            'use_seeding': use_seeding,
            'use_cache': self.use_cache,
            'batch_size': batch_size,
            'tts_model': self.tts.__class__.__name__,
            'ssl_model_config': self.sv_model.config_path,
            'ssl_model_checkpoint': self.sv_model.checkpoint_name,
            'verification_threshold': self.verification_threshold,
            'eer_method': self.eer_method,
            'results': results
        }
        
        return attack_summary

def main():
    parser = argparse.ArgumentParser(description='VoxCeleb Speech Verification Attack Script using SSL Models')
    parser.add_argument('--ssl_config', type=str, required=True,
                       help='Path to SSL model config file (required)')
    parser.add_argument('--ssl_checkpoint', type=str, default='model_latest.pt',
                       help='SSL model checkpoint name (default: model_latest.pt)')
    parser.add_argument('--tts', type=str, default='F5TTS', 
                       help='TTS model to use (F5TTS recommended for voice cloning)')
    parser.add_argument('--voxceleb_root', type=str, default='/media/volume/AudioUnlearnData1/sslsv/data/',
                       help='Root directory of VoxCeleb test data')
    parser.add_argument('--protocol_file', type=str, default='voxceleb1_test_O',
                       help='VoxCeleb1-O protocol file (relative to SSL model base_path, e.g. "voxceleb1_test_O")')
    parser.add_argument('--transcript_file', type=str, default='voxceleb_transcripts_test.csv',
                       help='CSV file with VoxCeleb transcripts (relative to SSL model base_path)')
    parser.add_argument('--output_dir', type=str, default='sv_ssl_attack_results_05072025_16khz',
                       help='Output directory for results')
    parser.add_argument('--cache_dir', type=str, default='cache',
                       help='Cache directory for generated audio files')
    parser.add_argument('--max_trials', type=int, default=None,
                       help='Maximum number of trials to process (default: None for all trials)')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='Batch size for processing trials (default: 10000)')
    parser.add_argument('--use_seeding', action='store_true', default=True,
                       help='Use deterministic seeding (default: True)')
    parser.add_argument('--no_seeding', dest='use_seeding', action='store_false',
                       help='Disable deterministic seeding')
    parser.add_argument('--seed_value', type=int, default=16,
                       help='Seed value for deterministic seeding (default: 16)')
    parser.add_argument('--use_cache', action='store_true', default=True,
                       help='Use caching for generated audio (default: True)')
    parser.add_argument('--no_cache', dest='use_cache', action='store_false',
                       help='Disable caching for generated audio')
    parser.add_argument('--eer_method', type=str, default='discrete', choices=['discrete', 'interpolation'],
                       help='EER threshold computation method (discrete: SSL framework style, interpolation: more precise)')
    args = parser.parse_args()
    
    # Setup directories and logging
    log_dir = setup_directories(args.output_dir, args.cache_dir)
    setup_logging(log_dir)
    
    # Log experiment start
    experiment_start_time = datetime.datetime.now()
    logger.info(f"=== VoxCeleb SSL Speech Verification Attack Started ===")
    logger.info(f"Start time: {experiment_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Cache directory: {args.cache_dir}")
    
    # Initialize SSL speech verification model
    logger.info(f"Loading SSL speech verification model from config: {args.ssl_config}")
    logger.info(f"Using checkpoint: {args.ssl_checkpoint}")
    sv_model = SSLSpeechVerificationModel(
        config_path=args.ssl_config,
        checkpoint_name=args.ssl_checkpoint
    )
    
    # Initialize TTS model
    logger.info(f"Loading TTS model: {args.tts}")
    tts_cls = TTS_MODELS_REGISTER[args.tts]
    tts = tts_cls(use_cache=args.use_cache, cache_dir=args.cache_dir)
    
    # Create attacker
    attacker = VoxCelebSSLAttacker(
        sv_model=sv_model,
        tts=tts,
        voxceleb_root=args.voxceleb_root,
        protocol_file=args.protocol_file,
        target_transcript_file=args.transcript_file,
        use_cache=args.use_cache,
        cache_dir=args.cache_dir,
        use_seeding=args.use_seeding,
        seed_value=args.seed_value,
        eer_method=args.eer_method
    )
    
    # Generate output filename with parameters
    max_trials_str = str(args.max_trials) if args.max_trials is not None else 'all'
    seeding_suffix = "seeded" if args.use_seeding else "unseeded"
    cache_suffix = "cached" if args.use_cache else "nocache"
    ssl_model_name = os.path.basename(args.ssl_config).replace('.yml', '').replace('config', 'ssl')
    output_file = os.path.join(args.output_dir, f'voxceleb_ssl_attack_results-{ssl_model_name}-{args.ssl_checkpoint.replace(".pt", "")}-{args.tts}-{max_trials_str}-{seeding_suffix}-{cache_suffix}.json')
    
    # Run attack
    logger.info(f"Starting VoxCeleb SSL speech verification attack (seeding: {args.use_seeding}, cache: {args.use_cache})...")
    attack_results = attacker.run_attack(max_trials=args.max_trials, use_seeding=args.use_seeding, batch_size=args.batch_size, output_file=output_file)
    
    # Save results
    if attack_results:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(attack_results, f, ensure_ascii=False, indent=4)
        
        # Display summary
        print("\n" + "="*80)
        print("VOXCELEB SSL SPEECH VERIFICATION ATTACK RESULTS")
        print("="*80)
        print(f"TTS Model: {attack_results['tts_model']}")
        print(f"SSL Model Config: {attack_results['ssl_model_config']}")
        print(f"SSL Model Checkpoint: {attack_results['ssl_model_checkpoint']}")
        print(f"Verification Threshold (EER): {attack_results['verification_threshold']:.4f}")
        print(f"EER Computation Method: {attack_results['eer_method']}")
        print(f"Using seeding: {attack_results['use_seeding']}")
        print(f"Using cache: {attack_results['use_cache']}")
        print(f"Batch size: {attack_results['batch_size']}")
        print(f"Total trials: {attack_results['total_trials']}")
        print(f"Processing errors: {attack_results['processing_errors']}")
        print(f"Successful trials: {attack_results['successful_trials']}")
        print(f"Same speaker trials: {attack_results['same_speaker_trials']}")
        print(f"Different speaker trials: {attack_results['diff_speaker_trials']}")
        print(f"Same speaker attack success rate: {attack_results['same_speaker_success_rate']:.2f}%")
        print(f"Different speaker attack success rate: {attack_results['diff_speaker_success_rate']:.2f}%")
        print(f"Overall attack success rate: {attack_results['overall_success_rate']:.2f}%")
        print(f"Processing time: {attack_results['processing_time_formatted']}")
        print(f"Results saved to: {output_file}")
        
        # Log experiment completion
        experiment_end_time = datetime.datetime.now()
        total_runtime = (experiment_end_time - experiment_start_time).total_seconds()
        logger.info("=== VoxCeleb SSL Speech Verification Attack Completed ===")
        logger.info(f"End time: {experiment_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total runtime: {total_runtime:.2f} seconds")
    else:
        logger.error("Attack failed - no results to save")

if __name__ == "__main__":
    main() 