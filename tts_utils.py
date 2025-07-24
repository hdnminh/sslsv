import os
import json
import time
import tomli
import struct
import hashlib
import librosa
import requests
import warnings
import torchaudio
import soundfile as sf
# from TTS.api import TTS
# from kokoro import KPipeline
from typing import Type, Dict
from dotenv import load_dotenv
from omegaconf import OmegaConf
from cached_path import cached_path
from importlib.resources import files
from f5_tts.model import DiT
import tqdm
from f5_tts.infer.utils_infer import (
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
    infer_process,
    preprocess_ref_audio_text,
    load_model,
    load_vocoder
)
from vars import f5tts_path
from utils import HiddenPrints

load_dotenv()
OPENAI_KEY = os.getenv('OPENAI_KEY')

warnings.simplefilter(action='ignore', category=FutureWarning)


def hash_to_int32(text):
    # Create a SHA-256 hash of the text
    hash_object = hashlib.sha256(text.encode())
    
    # Get the hash digest as bytes
    hash_bytes = hash_object.digest()
    
    # Take the first 4 bytes (32 bits) of the hash and convert them to an integer
    int32_value = struct.unpack('I', hash_bytes[:4])[0]
    
    return str(int32_value)

class TTSGenerator:

    def __init__(self, voice = None, cache_dir = "", use_cache = True):
        self.is_commercial = False
        self.cache_dir = cache_dir
        self.voice = voice
        self.use_cache = use_cache
        if voice:
            self.cache_dir = os.path.join(cache_dir, voice)

        # Only create cache directory if caching is enabled and cache_dir is not empty
        if self.use_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __call__(self, texts, **kwargs):
        """
        Generate audio with advanced caching support.
        
        Args:
            texts: List of texts to generate audio for
            **kwargs: Advanced caching options:
                - speaker_structure: Dict with 'speaker_id', 'video_id', 'audio_filename'
                - use_seeding: bool, whether to include seed in filename
                - seed_value: int, seed value to use (default 42)
                - cache_mode: str, 'simple' (default) or 'speaker_based' or 'custom'
                - custom_cache_paths: List of custom cache paths (same length as texts)
                - target_audio_rel_path: str, relative path for original fake audio lookup
                - target_audio_path: str, path to target audio for voice cloning
                - target_transcript: str, transcript of target audio (needed for F5TTS)
        """
        if isinstance(texts, str):
            texts = [texts]

        # Extract caching options
        cache_mode = kwargs.get('cache_mode', 'simple')
        use_seeding = kwargs.get('use_seeding', False)
        seed_value = kwargs.get('seed_value', 42)
        speaker_structure = kwargs.get('speaker_structure', {})
        custom_cache_paths = kwargs.get('custom_cache_paths', [])
        target_audio_rel_path = kwargs.get('target_audio_rel_path', None)
        target_audio_path = kwargs.get('target_audio_path', None)
        target_transcript = kwargs.get('target_transcript', None)
        original_fake_audios_path = kwargs.get('original_fake_audios_path', None)

        # Setup voice cloning if target audio is provided with the correct original transcript
        self._setup_voice_cloning(target_audio_path, target_transcript)


        audio_files = []
        cached_files = []
        pending_files = []
        
        for idx, text in enumerate(texts):
            # Determine cache path based on mode
            if cache_mode == 'custom' and idx < len(custom_cache_paths):
                cached_file = custom_cache_paths[idx]
            elif cache_mode == 'speaker_based':
                cached_file = self._get_speaker_based_cache_path(
                    text, speaker_structure, use_seeding, seed_value, 
                    target_transcript, target_audio_rel_path, original_fake_audios_path
                )
                if cached_file is None:  # Skip this sample (original fake audio not found)
                    audio_files.append(None)
                    continue
            else:  # simple mode (backward compatibility)
                hash_value = hash_to_int32(text)
                if use_seeding:
                    cached_file = os.path.join(self.cache_dir, f'{hash_value}_seed{seed_value}.wav')
                else:
                    cached_file = os.path.join(self.cache_dir, f'{hash_value}.wav')

            # Ensure cache directory exists (only if caching is enabled)
            if self.use_cache:
                os.makedirs(os.path.dirname(cached_file), exist_ok=True)

            if self.use_cache and os.path.exists(cached_file):
                cached_files.append((idx, cached_file))
            else:
                pending_files.append((idx, cached_file, text))

        if len(pending_files) == 0:
            audio_files = [d[1] if d is not None else None for d in cached_files]
        else:
            gen_files = self.generate_audio(texts = [d[2] for d in pending_files], 
                                            output_paths = [d[1] for d in pending_files])
            
            # Merge cached and generated files
            all_files = cached_files + [(u[0], v) for u, v in zip(pending_files, gen_files)]
            all_files = sorted([f for f in all_files if f[1] is not None], key=lambda x: x[0])
            
            # Create final list maintaining original order, with None for skipped samples
            audio_files = [None] * len(texts)
            for orig_idx, file_path in all_files:
                audio_files[orig_idx] = file_path

        return audio_files

    def _setup_voice_cloning(self, target_audio_path, target_transcript):
        """
        Setup voice cloning for TTS models that support it.
        Each TTS subclass can override this method for their specific setup.
        """
        if not target_audio_path or not os.path.exists(target_audio_path):
            return  # No voice cloning setup needed
            
        # Default implementation - subclasses should override
        # This method is called before generation to set up voice cloning
        pass

    def _get_speaker_based_cache_path(
        self, 
        text, 
        speaker_structure, 
        use_seeding, 
        seed_value, 
        target_transcript, 
        target_audio_rel_path,
        original_fake_audios_path
    ):
        """

        
        Returns None if this is an original transcript but original fake audio not found
        """
        hash_value = hash_to_int32(text)
        
        # Check if the target_transcript is the same as the original text and try to reuse existing fake audio
        if (target_transcript and text == target_transcript and 
            target_audio_rel_path and target_audio_rel_path != ''):
            
            rel_path = target_audio_rel_path
            parts = rel_path.split('/')
            if len(parts) >= 3:
                speaker_id, video_id, audio_filename = parts[-3], parts[-2], parts[-1]
                
                # Construct path to original fake audio
                seeding_mode = "seeded" if use_seeding else "unseeded"
                tts_method = self.__class__.__name__
                original_fake_base = f"{original_fake_audios_path}/{tts_method}/{seeding_mode}"
                
                if use_seeding:
                    name, ext = os.path.splitext(audio_filename)
                    seeded_filename = f"{name}_seed{seed_value}{ext}"
                    original_fake_audio_path = os.path.join(original_fake_base, speaker_id, video_id, seeded_filename)
                else:
                    original_fake_audio_path = os.path.join(original_fake_base, speaker_id, video_id, audio_filename)
                
                # Check if original fake audio exists
                if os.path.exists(original_fake_audio_path):
                    return original_fake_audio_path
                else:
                    print(f"Original fake audio before real attacknot found: {original_fake_audio_path}")
                    return None
        
        # Use speaker-based caching for perturbed transcripts or when original not available during real attack
        if target_audio_rel_path:
            rel_path = target_audio_rel_path
            parts = rel_path.split('/')
            if len(parts) >= 3:
                speaker_id, video_id, audio_filename = parts[-3], parts[-2], parts[-1]
                
                # Create cache path with speaker structure and hash differentiation
                seeding_mode = "seeded" if use_seeding else "unseeded"
                tts_method = self.__class__.__name__
                cache_root = f"{self.cache_dir}/{tts_method}/{seeding_mode}"
                
                # Create subdirectory for this audio file and use hash to differentiate variants
                name, ext = os.path.splitext(audio_filename)
                audio_subdir = os.path.join(cache_root, speaker_id, video_id, name)
                
                if use_seeding:
                    variant_filename = f"{hash_value}_{name}_seed{seed_value}{ext}"
                else:
                    variant_filename = f"{hash_value}_{name}{ext}"
                    
                return os.path.join(audio_subdir, variant_filename)
        
        # Fallback to hash-based naming
        seeding_mode = "seeded" if use_seeding else "unseeded"
        tts_method = self.__class__.__name__
        cache_root = f"{self.cache_dir}/{tts_method}/{seeding_mode}"
        
        if use_seeding:
            filename = f'{hash_value}_seed{seed_value}.wav'
        else:
            filename = f'{hash_value}.wav'
            
        return os.path.join(cache_root, filename)

    def generate_audio(self, texts, output_paths):
        raise NotImplementedError("Subclasses must override generate_audio()")

    def update_voice(self, new_voice):
        if self.voice is None:
            self.cache_dir = os.path.join(self.cache_dir, new_voice)
        else:
            self.cache_dir = self.cache_dir.replace(self.voice, new_voice)
        # Only create cache directory if caching is enabled and cache_dir is not empty
        if self.use_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.voice = new_voice

    def get_voice_profiles(self):
        raise NotImplementedError("Subclasses must override get_voice_profiles()")

class KokoroTTS(TTSGenerator):
    def __init__(self, voice='af_heart', sampling_rate=22050, cache_dir="", use_cache=True):
        super().__init__(voice, cache_dir, use_cache)
        self.tts_pipeline = KPipeline(lang_code='a')
        self.available_voices = ['af_heart', 'am_adam', 'bf_lily', 'bm_george']
        self.voice = voice
        self.sampling_rate = sampling_rate
        
        # Validate voice
        if voice not in self.available_voices:
            raise ValueError(f"Voice '{voice}' not supported. Available voices: {self.available_voices}")

    def _setup_voice_cloning(self, target_audio_path, target_transcript):
        """KokoroTTS doesn't support voice cloning - uses predefined voices only"""
        if target_audio_path:
            # Log warning that voice cloning is not supported
            pass  # Already logged in main attack code

    def get_voice_profiles(self):
        # Return available predefined voices (KokoroTTS doesn't support voice cloning)
        return self.available_voices
    
    def set_voice(self, voice):
        """Set voice from available predefined voices"""
        if voice not in self.available_voices:
            raise ValueError(f"Voice '{voice}' not supported. Available voices: {self.available_voices}")
        self.voice = voice
        
    def update_voice(self, voice):
        """Update voice using predefined voice name"""
        self.set_voice(voice)
        super().update_voice(voice)

    def generate_audio(self, texts, output_paths):
        if isinstance(texts, list):
            texts = "\n\n".join(texts)
        if isinstance(output_paths, str):
            output_paths = [output_paths]

        generator = self.tts_pipeline(
            texts, voice=self.voice, speed=1, split_pattern=r'\n+'
        )

        for i, (gs, ps, audio) in enumerate(generator):
            sf.write(output_paths[i], audio, self.sampling_rate)

        return output_paths


class CoquiTTS(TTSGenerator):

    def __init__(self, voice=None, cache_dir="", use_cache=True):
        super().__init__(voice, cache_dir, use_cache)
        # Initialize with no speaker_wav - will be set dynamically for voice cloning
        self.speaker_wav = None
        self._original_speaker_wav = None  # Store original for restoration
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to('cuda').eval()

    def _setup_voice_cloning(self, target_audio_path, target_transcript):
        """Setup voice cloning for CoquiTTS"""
        if target_audio_path and os.path.exists(target_audio_path):
            # Store original for restoration later
            self._original_speaker_wav = self.speaker_wav
            self.set_speaker_wav(target_audio_path)

    def _restore_voice_cloning(self):
        """Restore original voice cloning setup"""
        if hasattr(self, '_original_speaker_wav'):
            self.speaker_wav = self._original_speaker_wav

    def get_voice_profiles(self):
        # Return empty list since we use dynamic voice cloning
        return []

    def generate_audio(self, texts, output_paths):
        if self.speaker_wav is None:
            raise ValueError("speaker_wav must be set before generating audio. Use set_speaker_wav() or temporarily set self.speaker_wav")
        
        with HiddenPrints():
            for text, output_path in zip(texts, output_paths):
                # Automatically handle long text by enabling sentence splitting
                split_sentences = len(text) > 250
                
                self.tts.tts_to_file(text=text, 
                speaker_wav=self.speaker_wav, 
                language="en", 
                file_path=output_path, 
                split_sentences=split_sentences)

        return output_paths

    def set_speaker_wav(self, speaker_wav_path):
        """Set the speaker wav file for voice cloning"""
        self.speaker_wav = speaker_wav_path
        
    def update_voice(self, voice_path):
        """Update voice using any audio file path for voice cloning"""
        self.speaker_wav = voice_path
        super().update_voice(voice_path)
        self.voice = voice_path


class F5TTSGenerator(TTSGenerator):
    def __init__(self, voice=None, cache_dir="", use_cache=True):
        super().__init__(voice, cache_dir, use_cache)
        
        # Initialize with no reference audio/text - will be set dynamically for voice cloning
        self.ref_audio = None
        self.ref_text = None
        self._original_ref_audio = None  # Store original for restoration
        self._original_ref_text = None

        model_cls = DiT
        model_cfg = str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml"))
        model_cfg = OmegaConf.load(model_cfg).model.arch
        ckpt_file = str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors", cache_dir = f5tts_path))
        # model_cfg = str(files("f5_tts").joinpath("configs/F5TTS_Base.yaml"))
        # model_cfg = OmegaConf.load(model_cfg).model.arch
        # ckpt_file = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors", cache_dir = f5tts_path))
        self.vocoder_name = "vocos"
        self.ema_model = load_model(model_cls, 
                                    model_cfg, 
                                    ckpt_file, 
                                    mel_spec_type=self.vocoder_name, 
                                    vocab_file="")
        self.ema_model.eval()

        self.vocoder = load_vocoder(vocoder_name=self.vocoder_name, 
                                    is_local=False, 
                                    local_path="../checkpoints/vocos-mel-24khz")
        self.vocoder.eval()

        self.sampling_rate = 16000

    def _setup_voice_cloning(self, target_audio_path, target_transcript):
        """Setup voice cloning for F5TTS"""
        if target_audio_path and target_transcript and os.path.exists(target_audio_path):
            # Store original for restoration later
            self._original_ref_audio = self.ref_audio
            self._original_ref_text = self.ref_text
            self.set_reference_audio(target_audio_path, target_transcript)

    def _restore_voice_cloning(self):
        """Restore original voice cloning setup"""
        if hasattr(self, '_original_ref_audio'):
            self.ref_audio = self._original_ref_audio
            self.ref_text = self._original_ref_text

    def get_voice_profiles(self):
        # Return empty list since we use dynamic voice cloning
        return []

    def set_reference_audio(self, ref_audio_path, ref_text):
        """Set the reference audio and text for voice cloning"""
        self.ref_audio = ref_audio_path
        self.ref_text = ref_text

    def update_voice(self, voice_path, ref_text=None):
        """Update voice using any audio file path for voice cloning"""
        self.ref_audio = voice_path
        if ref_text is not None:
            self.ref_text = ref_text
        super().update_voice(voice_path)
        self.voice = voice_path

    def single_infer(self, gen_text, wave_path):
        if self.ref_audio is None or self.ref_text is None:
            raise ValueError("ref_audio and ref_text must be set before generating audio. Use set_reference_audio()")
            
        # WARNING: F5TTS has a known bug where it doesn't properly concatenate audio batches
        # for long text, resulting in truncated audio output. For reliable long text generation,
        # use CoquiTTS instead.

        length_ref_audio = librosa.get_duration(filename=self.ref_audio) 
        if length_ref_audio > 12:
            self.ref_audio, self.ref_text = preprocess_ref_audio_text(
                                                ref_audio_orig = self.ref_audio, 
                                                ref_text = ""
                                            )

        result = infer_process( 
            self.ref_audio,
            self.ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
            mel_spec_type=self.vocoder_name,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            progress=tqdm
        )
        
        # Handle different return formats from infer_process
        if len(result) == 3:
            audio_wave, final_sample_rate, _ = result
        elif len(result) == 2:
            audio_wave, final_sample_rate = result
        else:
            audio_wave = result[0]
            final_sample_rate = 24000  # F5TTS default

        # Ensure audio_wave is not None
        if audio_wave is None:
            raise ValueError("Failed to generate audio - infer_process returned None")

        if final_sample_rate != self.sampling_rate:
            audio_wave = librosa.resample(audio_wave, orig_sr=final_sample_rate, target_sr=self.sampling_rate)
            
        # Save audio file correctly using soundfile
        sf.write(wave_path, audio_wave, self.sampling_rate)

        return

    def generate_audio(self, texts, output_paths):
        if self.ref_audio is None or self.ref_text is None:
            raise ValueError("ref_audio and ref_text must be set before generating audio. Use set_reference_audio()")
            
        with HiddenPrints():
            for text, output_path in zip(texts, output_paths):
                self.single_infer(text, output_path)

        return output_paths

def openai_tts(transcript, out_path, voice = "shimmer", model = "tts-1"):
    url = "https://api.openai.com/v1/audio/speech"

    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "input": transcript,
        "voice": voice,
        'speed': 1
    }

    r = requests.post(url, headers=headers, data=json.dumps(data))
    
    audio_content = r.content
        
    with open(out_path, 'wb') as f:
        f.write(audio_content)

    return out_path

class OpenAITTS(TTSGenerator):
    def __init__(self, voice='alloy', cache_dir="", use_cache=True):
        super().__init__(voice, cache_dir, use_cache)
        # All available OpenAI TTS voices
        self.available_voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
        self.voice = voice
        self.sampling_rate = 22050
        self.is_commercial = True
        
        # Validate voice
        if voice not in self.available_voices:
            raise ValueError(f"Voice '{voice}' not supported. Available voices: {self.available_voices}")

    def _setup_voice_cloning(self, target_audio_path, target_transcript):
        """OpenAI TTS doesn't support voice cloning - uses predefined voices only"""
        if target_audio_path:
            # Log warning that voice cloning is not supported
            pass  # Already logged in main attack code

    def get_voice_profiles(self):
        # Return available predefined voices (OpenAI TTS doesn't support voice cloning)
        return self.available_voices
    
    def set_voice(self, voice):
        """Set voice from available OpenAI predefined voices"""
        if voice not in self.available_voices:
            raise ValueError(f"Voice '{voice}' not supported. Available voices: {self.available_voices}")
        self.voice = voice
        
    def update_voice(self, voice):
        """Update voice using predefined OpenAI voice name"""
        self.set_voice(voice)
        super().update_voice(voice)
    
    def generate_audio(self, texts, output_paths):
        for text, output_path in zip(texts, output_paths):
            openai_tts(transcript=text, out_path=output_path, voice=self.voice)
            audio, orig_sr = torchaudio.load(output_path)
            if orig_sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(orig_sr, self.sampling_rate)
                audio = resampler(audio)

            torchaudio.save(output_path, audio, self.sampling_rate)
    
        return output_paths


TTS_MODELS_REGISTER: Dict[str, Type[TTSGenerator]] = {
    "KokoroTTS": KokoroTTS,
    "F5TTS": F5TTSGenerator,
    "CoquiTTS": CoquiTTS,
    "OpenAITTS": OpenAITTS
}