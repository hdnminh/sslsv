#!/usr/bin/env python3
"""
VoxCeleb Speech Verification Attack Script
Performs voice cloning attacks on speech verification systems using VoxCeleb1-O protocol
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

# # Filter out warnings
# warnings.filterwarnings("ignore", message=".*cumsum_cuda_kernel does not have a deterministic implementation.*")
# warnings.filterwarnings("ignore", message=".*text length exceeds the character limit.*")

# STT no longer needed - using pre-loaded transcripts
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
        os.path.join(log_dir, "original_simple_attack_F5TTS.log"),
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
    
    def __init__(self, model_name: str = 'b6', dataset: str = 'vox2', train_type: str = 'ft_lm'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.dataset = dataset
        self.train_type = train_type
        self.model: Optional[torch.nn.Module] = None
        self.load_model()
        
    def load_model(self):
        """Load pretrained ReDimNet model"""
        try:
            logger.info(f"Loading ReDimNet model: {self.model_name}, dataset: {self.dataset}, train_type: {self.train_type}")
            loaded_model = torch.hub.load('IDRnD/ReDimNet', 'ReDimNet', 
                                      model_name=self.model_name, 
                                      train_type=self.train_type, 
                                      dataset=self.dataset,
                                      trust_repo=True)
            if loaded_model is not None:
                self.model = loaded_model  # type: ignore
                assert self.model is not None  # Help type checker
                self.model.to(self.device)
                self.model.eval()
                logger.info("Model loaded successfully")
            else:
                raise ValueError("Model loading returned None")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def extract_embedding(self, audio: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract embedding from audio using the model"""
        try:
            if self.model is None:
                logger.error("Model is not loaded")
                return None
                
            with torch.no_grad():
                # Add batch dimension
                audio_batch = audio.unsqueeze(0).to(self.device)
                
                # Get embedding
                embedding = self.model(audio_batch)
                
                # Normalize embedding
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                
                return embedding.squeeze(0).cpu()
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            return None
    
    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between two embeddings"""
        return (torch.dot(emb1, emb2).item() + 1) / 2  # Convert from [-1, 1] to [0, 1]

class VoxCelebAttacker:
    """
    VoxCeleb speech verification attacker using voice cloning
    """
    
    def __init__(
        self, 
        sv_model: SpeechVerificationModel, 
        tts: TTSGenerator, 
        voxceleb_root: str = "/media/volume/AudioUnlearnData1/emnlp_audio_adversarial/data/test", 
        protocol_file: str = "/media/volume/AudioUnlearnData1/emnlp_audio_adversarial/data/data-list/vox1-O.txt",
        target_transcript_file: str = "/media/volume/AudioUnlearnData1/emnlp_audio_adversarial/data/voxceleb_transcripts_test.csv", 
        use_cache: bool = True, 
        cache_dir: str = "cache",
        use_seeding: bool = True,
        seed_value: int = 75
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
        # Load transcripts from CSV
        self.target_transcript_map = self._load_all_transcripts()
        
        # Statistics
        self.total_trials = 0
        self.successful_attacks = 0
        self.failed_attacks = 0
        self.processing_errors = 0
        
    def _load_all_transcripts(self) -> Dict[str, str]:
        """Load transcripts from CSV file"""
        target_transcript_map = {}
        try:
            df = pd.read_csv(self.target_transcript_file)
            for _, row in df.iterrows():
                # Use audio_path as key (relative path from voxceleb root)
                audio_path = str(row['audio_path'])
                # Extract relative path from full path
                if audio_path.startswith('/'):
                    # Remove the voxceleb root prefix to get relative path
                    relative_path = audio_path.split('/test/')[-1] if '/test/' in audio_path else audio_path
                else:
                    relative_path = audio_path
                target_transcript_map[relative_path] = str(row['full_transcript'])
            logger.info(f"Loaded {len(target_transcript_map)} transcripts from {self.target_transcript_file}")
            return target_transcript_map
        except Exception as e:
            logger.error(f"Failed to load transcripts from {self.target_transcript_file}: {e}")
            return {}
        
    def load_protocol(self) -> List[Tuple[int, str, str]]:
        """Load VoxCeleb1-O test protocol"""
        trials = []
        try:
            with open(self.protocol_file, 'r') as f:
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
            logger.error(f"Failed to load protocol from {self.protocol_file}: {e}")
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

            # If not found, try to match by filename
            # audio_filename = os.path.basename(audio_path)
            # for path, transcript in self.target_transcript_map.items():
            #     if os.path.basename(path) == audio_filename:
            #         return transcript
            
            logger.warning(f"No transcript found for {audio_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to get transcript for {audio_path}: {e}")
            return None
    
    def get_audio_path(self, audio_path: str) -> str:
        """Get full path to audio file for voice cloning"""
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
        # Use instance cache setting if not provided

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
                temp_dir = tempfile.mkdtemp(prefix="nocache_attack_")
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
                # original_speaker_wav = getattr(self.tts, 'speaker_wav', None)
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
                # finally:
                #     # Restore original speaker_wav
                #     if original_speaker_wav is not None:
                #         self.tts.speaker_wav = original_speaker_wav
                    
            elif isinstance(self.tts, F5TTSGenerator):
                # Use voice cloning with the new cleaner interface
                # original_ref_audio = getattr(self.tts, 'ref_audio', None)
                # original_ref_text = getattr(self.tts, 'ref_text', None)
                
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
                # finally:
                #     # Restore original reference audio/text
                #     if original_ref_audio is not None:
                #         self.tts.ref_audio = original_ref_audio
                #     if original_ref_text is not None:
                #         self.tts.ref_text = original_ref_text
                    
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
            
            # Extract embeddings
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
    
    def create_voice_profile(self, target_audio_abs: str, output_path: str) -> bool:
        """Create voice profile from target audio for voice cloning"""
        try:
            if os.path.exists(output_path):
                return True
            # Copy the audio file to voice profiles directory for TTS use
            audio, sr = librosa.load(target_audio_abs, sr=16000)  # CoquiTTS typically uses 22050
            sf.write(output_path, audio, sr)
            return True
        except Exception as e:
            logger.error(f"Failed to create voice profile from {target_audio_abs}: {e}")
            return False
    
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
                
                # ref_audio_abs = os.path.join(self.voxceleb_root, ref_audio_rel)
                original_similarity = self.compute_similarity(ref_audio_rel, target_audio_data)
                if original_similarity is None:
                    result['error'] = 'Failed to compute original similarity'
                    return result
                
                result['original_similarity'] = original_similarity
                
                # Step 3: Generate fake audio using voice cloning with seeding
                target_audio_abs = os.path.join(self.voxceleb_root, target_audio_rel)
                # parts = target_audio_rel.split('/')
                # if len(parts) >= 3:
                #     speaker_id, video_id, audio_filename = parts[-3], parts[-2], parts[-1]
                # voice_profile_path = f"{self.cache_dir}/voice_profiles/{speaker_id}/{video_id}/{audio_filename}"
                # os.makedirs(os.path.dirname(voice_profile_path), exist_ok=True)
                # if not self.create_voice_profile(target_audio_abs, voice_profile_path):
                #     result['error'] = 'Failed to create voice profile'
                #     return result
                #voice_profile_path makes the target voice to sampling rate 22050 and use it for voice cloning without any other changes of the original audio
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
                threshold = 0.68  # Typical verification threshold
                
                if label == 1:
                    # Same speaker: attack succeeds if fake similarity remains high
                    result['success'] = fake_similarity >= threshold
                else:
                    # Different speakers: attack succeeds if fake similarity becomes high
                    result['success'] = fake_similarity >= threshold and fake_similarity > original_similarity
                
                # Note: We keep the fake audio in cache for future use, no cleanup needed
                    
            finally:
                # Clean up voice profile
                # if os.path.exists(voice_profile_path):
                #     os.remove(voice_profile_path)
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
            "sv_model": f"{self.sv_model.model_name}-{self.sv_model.dataset}-{self.sv_model.train_type}",
            "tts_model": self.tts.__class__.__name__,
            "attack_strategy": "voice_cloning",
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
            'sv_model': f"{self.sv_model.model_name}-{self.sv_model.dataset}-{self.sv_model.train_type}",
            'results': results
        }
        
        return attack_summary

def main():
    parser = argparse.ArgumentParser(description='VoxCeleb Speech Verification Attack Script')
    parser.add_argument('--sv_model', type=str, default='b6', 
                       help='ReDimNet model to attack (M, S, b0-b6)')
    parser.add_argument('--dataset', type=str, default='vox2',
                       help='ReDimNet dataset (vb2, vox2, vb2+vox2+cnc)')
    parser.add_argument('--train_type', type=str, default='ft_lm',
                       help='ReDimNet training type (ptn, ft_lm, ft_mix)')
    parser.add_argument('--tts', type=str, default='F5TTS', 
                       help='TTS model to use (F5TTS recommended for voice cloning)')
    parser.add_argument('--voxceleb_root', type=str, default='/media/volume/AudioUnlearnData1/emnlp_audio_adversarial/data/test',
                       help='Root directory of VoxCeleb test data')
    parser.add_argument('--protocol_file', type=str, default='data/data-list/vox1-O.txt',
                       help='VoxCeleb1-O protocol file')
    parser.add_argument('--transcript_file', type=str, default='data/voxceleb_transcripts_test.csv',
                       help='CSV file with VoxCeleb transcripts')
    parser.add_argument('--output_dir', type=str, default='sv_simple_attack_results_05072025_16khz',
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
    args = parser.parse_args()
    
    # Setup directories and logging
    log_dir = setup_directories(args.output_dir, args.cache_dir)
    setup_logging(log_dir)
    
    # Log experiment start
    experiment_start_time = datetime.datetime.now()
    logger.info(f"=== VoxCeleb Speech Verification Attack Started ===")
    logger.info(f"Start time: {experiment_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Cache directory: {args.cache_dir}")
    
    # Initialize speech verification model
    logger.info(f"Loading speech verification model: {args.sv_model}-{args.dataset}-{args.train_type}")
    sv_model = SpeechVerificationModel(
        model_name=args.sv_model,
        dataset=args.dataset,
        train_type=args.train_type
    )
    
    # Initialize TTS model
    logger.info(f"Loading TTS model: {args.tts}")
    tts_cls = TTS_MODELS_REGISTER[args.tts]
    tts = tts_cls(use_cache=args.use_cache, cache_dir=args.cache_dir)
    
    # Create attacker
    attacker = VoxCelebAttacker(
        sv_model=sv_model,
        tts=tts,
        voxceleb_root=args.voxceleb_root,
        protocol_file=args.protocol_file,
        target_transcript_file=args.transcript_file,
        use_cache=args.use_cache,
        cache_dir=args.cache_dir,
        use_seeding=args.use_seeding,
        seed_value=args.seed_value
    )
    
    # Generate output filename with parameters
    max_trials_str = str(args.max_trials) if args.max_trials is not None else 'all'
    seeding_suffix = "seeded" if args.use_seeding else "unseeded"
    cache_suffix = "cached" if args.use_cache else "nocache"
    output_file = os.path.join(args.output_dir, f'voxceleb_attack_results-{args.sv_model}-{args.dataset}-{args.train_type}-{args.tts}-{max_trials_str}-{seeding_suffix}-{cache_suffix}.json')
    
    # Run attack
    logger.info(f"Starting VoxCeleb speech verification attack (seeding: {args.use_seeding}, cache: {args.use_cache})...")
    attack_results = attacker.run_attack(max_trials=args.max_trials, use_seeding=args.use_seeding, batch_size=args.batch_size, output_file=output_file)
    
    # Save results
    if attack_results:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(attack_results, f, ensure_ascii=False, indent=4)
        
        # Display summary
        print("\n" + "="*80)
        print("VOXCELEB SPEECH VERIFICATION ATTACK RESULTS")
        print("="*80)
        print(f"TTS Model: {attack_results['tts_model']}")
        print(f"SV Model: {attack_results['sv_model']}")
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
        logger.info("=== VoxCeleb Speech Verification Attack Completed ===")
        logger.info(f"End time: {experiment_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total runtime: {total_runtime:.2f} seconds")
    else:
        logger.error("Attack failed - no results to save")

if __name__ == "__main__":
    main() 