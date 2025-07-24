# SSL Speech Verification Attack with TextAttack Framework

This script (`attack_sv_vox1_O_pwws_ssl.py`) combines the power of **SSL (Self-Supervised Learning) models** with the **TextAttack framework** to perform sophisticated adversarial attacks on speech verification systems.

## 🔄 What This Script Does

1. **Loads SSL Models**: Uses self-supervised speech verification models instead of supervised ones
2. **Text Perturbation**: Leverages TextAttack to generate semantically similar but adversarial text
3. **Voice Cloning**: Converts perturbed text into fake audio using TTS with voice cloning
4. **Attack Evaluation**: Measures if the fake audio can fool the SSL speech verification system

## 🎯 Key Features

### ✨ SSL Model Integration
- **Automatic SSL Model Loading**: Uses the SSL framework's `load_models` utility
- **Dynamic EER Threshold**: Automatically computes optimal verification thresholds
- **SSL Framework Compatibility**: Works with SSL model configs and checkpoints
- **Embedding Extraction**: Uses SSL models for audio embedding extraction

### 🧠 TextAttack Framework
- **PWWS Attack**: Uses Probability Weighted Word Saliency for text perturbations
- **Semantic Preservation**: SBERT constraints ensure perturbed text maintains meaning
- **Goal Function**: Maximizes similarity between reference and generated fake audio
- **Attack Logging**: Comprehensive logging of attack progress and results

### 🎵 Voice Cloning Integration
- **F5TTS Support**: Advanced voice cloning with reference audio
- **CoquiTTS Support**: Alternative TTS with voice cloning capabilities
- **Deterministic Seeding**: Reproducible audio generation
- **Smart Caching**: Efficient caching of generated audio files

## 🚀 Usage

### Basic Usage
```bash
python attack_sv_vox1_O_pwws_ssl.py \
    --ssl_config "models/tests/simclr/simclr_e-ecapa/ssps_kmeans_25k_uni-1/config.yml" \
    --ssl_checkpoint "model_avg.pt" \
    --tts "F5TTS" \
    --max_examples 50 \
    --eer_method "interpolation"
```

### Run Example
```bash
python example_ssl_textattack.py
```

## 📋 Required Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--ssl_config` | Path to SSL model config file | **Required** |
| `--ssl_checkpoint` | SSL model checkpoint name | `model_latest.pt` |

## ⚙️ Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--eer_method` | EER computation method (`discrete`, `interpolation`) | `discrete` |
| `--tts` | TTS model to use | `F5TTS` |
| `--max_examples` | Maximum examples to attack | `None` (all) |
| `--batch_size` | Batch size for processing | `50` |
| `--similarity_threshold` | Attack success threshold | `0.7` |
| `--use_seeding` | Enable deterministic seeding | `True` |
| `--seed_value` | Seed value for reproducibility | `16` |
| `--voxceleb_root` | VoxCeleb audio files root directory | `/media/volume/AudioUnlearnData1/sslsv/data/` |

## 🔧 Configuration Files

### SSL Model Config
- **Format**: YAML configuration file from SSL framework
- **Location**: Usually in `models/` directory with SSL training configs
- **Example**: `models/tests/simclr/simclr_e-ecapa/ssps_kmeans_25k_uni-1/config.yml`

### Dataset Path
- **Format**: JSON file with filtered speaker verification pairs
- **Default**: `sv_simple_attack_results_05072025_16khz/all-filtered-cases_0.63.json`
- **Structure**: Contains `target_transcript`, `ref_audio_rel`, `target_audio_rel` fields

## 📊 Output Format

The script generates comprehensive JSON results with:

```json
{
  "ssl_model_config": "path/to/config.yml",
  "ssl_model_checkpoint": "model_avg.pt",
  "TTS_model": "F5TTSGenerator",
  "attacker": "pwws",
  "verification_threshold": 0.2943,
  "eer_method": "interpolation",
  "attack_strategy": "maximize_similarity_ssl",
  "results": [
    {
      "speaker_id": "id10270",
      "video_id": "x6uYqmx31kE", 
      "audio_filename": "00001.wav",
      "original_text": "Hello world",
      "original_similarity": 0.45,
      "perturbed_text": "Hi world",
      "perturbed_similarity": 0.78,
      "similarity_improvement": 0.33,
      "verification_threshold": 0.2943
    }
  ]
}
```

## 🆚 Comparison with Other Scripts

| Feature | `original_simple_attack_f5tts_16khz_ssl.py` | `attack_sv_vox1_O_pwws.py` | `attack_sv_vox1_O_pwws_ssl.py` |
|---------|-------------------------------------|---------------------------|----------------------------|
| **Models** | SSL Models | Supervised Models | **SSL Models** |
| **Text Perturbation** | ❌ None | ✅ TextAttack | ✅ **TextAttack** |
| **Voice Cloning** | ✅ F5TTS/CoquiTTS | ✅ F5TTS/CoquiTTS | ✅ **F5TTS/CoquiTTS** |
| **EER Threshold** | ✅ Auto-computed | ❌ Fixed | ✅ **Auto-computed** |
| **Framework** | Custom | TextAttack | **TextAttack + SSL** |
| **Attack Type** | Voice cloning only | Text perturbation + TTS | **Text perturbation + Voice cloning** |

## 🎯 Key Advantages

1. **🧠 Advanced SSL Models**: Leverages state-of-the-art self-supervised learning for speech verification
2. **📝 Intelligent Text Attacks**: Uses TextAttack's sophisticated perturbation strategies
3. **🎵 Realistic Voice Cloning**: Combines text perturbations with voice cloning for stronger attacks
4. **⚡ Automatic Thresholding**: No need to manually tune verification thresholds
5. **🔄 Reproducible Results**: Deterministic seeding ensures consistent results
6. **📊 Comprehensive Logging**: Detailed attack progress and result tracking

## 🔬 Research Applications

This script is particularly useful for:

- **Linguistic Sensitivity Analysis**: Study how SSL models respond to text perturbations
- **Robustness Testing**: Evaluate SSL model resilience against combined text+audio attacks  
- **Attack Comparison**: Compare effectiveness of different attack strategies
- **Model Evaluation**: Assess SSL model security across different speakers and content

## 🛠️ Implementation Details

### SSL Model Integration
- Uses `notebooks.notebooks_utils.load_models()` for consistent SSL model loading
- Implements `SSLSpeechVerificationModel` wrapper for TextAttack compatibility
- Automatic EER threshold computation using `CosineSVEvaluation`

### TextAttack Wrapper
- `SSLSpeechVerificationPipelineWrapper` extends `ModelWrapper` for TextAttack
- Handles SSL-specific audio loading and embedding extraction
- Returns similarity scores in TextAttack-compatible format `[match_prob, non_match_prob]`

### Audio Processing Pipeline
1. **Reference Audio**: Loads and extracts SSL embedding from reference audio
2. **Text Perturbation**: TextAttack generates semantically similar adversarial text
3. **TTS Generation**: Converts perturbed text to fake audio with voice cloning
4. **Similarity Computation**: SSL model computes similarity between reference and fake audio
5. **Attack Evaluation**: Compares similarity against EER threshold for attack success

## 📝 Example Workflow

```bash
# 1. Run the example script
python example_ssl_textattack.py

# 2. Check results
ls ssl_textattack_results_example/

# 3. Review attack logs  
tail ssl_textattack_results_example/logs/ssl_attack_F5TTS_attacker_pwws_seed_42.log

# 4. Analyze results JSON
python -c "import json; print(json.load(open('ssl_textattack_results_example/pwws_ssl_vox1-O_ssl_ssps_kmeans_25k_uni_1_model_avg_42.json'))['log_summary'])"
```

This script represents the state-of-the-art in combining self-supervised learning, text perturbations, and voice cloning for comprehensive speech verification attacks! 🚀 