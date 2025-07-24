# SSL Speech Verification Attack Script

This repository contains `original_simple_attack_f5tts_16khz_ssl.py`, an adaptation of the original supervised speech verification attack script to use **Self-Supervised Learning (SSL)** models instead of supervised ReDimNet models.

## Purpose

The main goal is to study **linguistic sensitivity in SSL-based speaker verification systems** by performing voice cloning attacks using TTS models.

## 🎯 Key Feature: Automatic Optimal Threshold

The SSL attack script **automatically computes the EER (Equal Error Rate) threshold** for your specific SSL model instead of using a hardcoded threshold. This ensures accurate attack evaluation tailored to each SSL model's characteristics.

- **Smart Threshold Selection**: Computes the optimal threshold where False Accept Rate = False Reject Rate
- **Model-Specific**: Each SSL model gets its own threshold based on its performance characteristics  
- **Research Accuracy**: Uses the threshold that provides the most balanced and meaningful attack results
- **Dual Methods**: Choose between discrete (SSL framework style) or interpolation (more precise) methods

### EER Computation Methods

| Method | Description | Accuracy | Use Case |
|--------|-------------|----------|----------|
| **`discrete`** (default) | SSL framework style, uses actual score values | Good | SSL framework consistency |
| **`interpolation`** | Mathematical interpolation for exact EER point | Higher | Research requiring maximum precision |

**Usage:**
```bash
# Use interpolation method (more precise)
--eer_method interpolation

# Use discrete method (SSL framework consistent)  
--eer_method discrete
```

## Key Differences from Original Script

### Original Script (`original_simple_attack_f5tts_16khz.py`)
- Uses **supervised ReDimNet models** from torch.hub
- Loads models with specific architecture parameters (b6, vox2, ft_lm)
- Targets supervised speaker verification systems

### SSL Script (`original_simple_attack_f5tts_16khz_ssl.py`)
- Uses **Self-Supervised Learning (SSL) models** 
- Loads models from custom config files and checkpoints
- Targets SSL-based speaker verification systems
- Studies linguistic sensitivity in SSL representations

## Requirements

1. **SSL Model Configuration**: You need a valid SSL model config file and checkpoint
2. **TTS Models**: Same TTS models as original (F5TTS, CoquiTTS, KokoroTTS, OpenAITTS)
3. **VoxCeleb Dataset**: Same dataset structure as original script
4. **Dependencies**: All original dependencies plus SSL model utilities

## Usage

### Basic Usage

```bash
python original_simple_attack_f5tts_16khz_ssl.py \
    --ssl_config "models/your_ssl_model/config.yml" \
    --ssl_checkpoint "model_latest.pt" \
    --tts "F5TTS" \
    --max_trials 100
```

### Full Example

```bash
python original_simple_attack_f5tts_16khz_ssl.py \
    --ssl_config "models/tests/simclr/simclr_e-ecapa/ssps_kmeans_25k_uni-1/config.yml" \
    --ssl_checkpoint "model_avg.pt" \
    --tts "F5TTS" \
    --voxceleb_root "/path/to/voxceleb/test" \
    --protocol_file "voxceleb1_test_O" \
    --transcript_file "voxceleb_transcripts_test.csv" \
    --eer_method "interpolation" \
    --output_dir "ssl_attack_results" \
    --cache_dir "cache_ssl" \
    --max_trials 1000 \
    --batch_size 50 \
    --use_seeding \
    --seed_value 42 \
    --use_cache
```

### Example Script

Use the provided example script:

```bash
# List available SSL models
python example_ssl_attack.py --list-models

# Run example attack
python example_ssl_attack.py
```

## Command Line Arguments

### Required Arguments
- `--ssl_config`: Path to SSL model config file (YAML format)

### Optional Arguments
- `--ssl_checkpoint`: SSL model checkpoint name (default: model_latest.pt)
- `--tts`: TTS model (F5TTS, CoquiTTS, KokoroTTS, OpenAITTS)
- `--eer_method`: EER computation method (discrete, interpolation; default: discrete)
- `--max_trials`: Maximum number of trials (default: all)
- `--batch_size`: Batch size for processing (default: 10000)
- `--use_seeding/--no_seeding`: Enable/disable deterministic seeding
- `--seed_value`: Seed value for reproducibility (default: 16)
- `--use_cache/--no_cache`: Enable/disable audio caching
- `--output_dir`: Output directory for results
- `--cache_dir`: Cache directory for generated audio

## SSL Model Configuration

The script expects SSL models to follow the standard SSL loading pattern:

1. **Config File**: YAML configuration file defining model architecture and training parameters
2. **Checkpoint File**: PyTorch checkpoint containing trained model weights
3. **Model Structure**: Compatible with the SSL evaluation framework

Example config structure:
```yaml
model:
  arch: YourSSLModelArchitecture
  # ... other model parameters

dataset:
  base_path: /path/to/data
  # ... dataset parameters
```

## File Path Structure

The script follows the SSL framework's file loading pattern:

### Protocol Files
- **Relative to `base_path`**: Protocol files are loaded relative to the SSL model's `dataset.base_path`
- **Automatic path resolution**: The script tries multiple locations:
  1. `{base_path}/{protocol_file}` (e.g., `/data/voxceleb1_test_O`)
  2. `{base_path}/{protocol_file}.txt` (with .txt extension)
  3. `{base_path}/trials/{protocol_file}` (in trials subdirectory)
  4. Fallback to hardcoded paths if needed

### Transcript Files
- **Also relative to `base_path`**: Transcript CSV files follow the same pattern
- **Automatic discovery**: The script searches for the file in multiple locations

### Example File Structure
```
/path/to/ssl/data/          # This is your base_path
├── voxceleb1_test_O        # Protocol file (no extension)
├── voxceleb1_test_E        # Other protocol files
├── voxceleb_transcripts_test.csv  # Transcript file
└── trials/                 # Alternative location
    ├── voxceleb1_test_O
    └── voxceleb1_test_E.txt
```

### Configuration Tips
1. **Check your SSL model's `base_path`**: Look at your SSL config file's `dataset.base_path`
2. **Place files correctly**: Put protocol and transcript files in the `base_path` directory
3. **Use relative names**: Use filenames like `"voxceleb1_test_O"` instead of full paths
4. **The script will tell you**: If files aren't found, the script lists all locations it tried

## Output

The script generates:

1. **JSON Results File**: Detailed attack results with metrics
2. **Log Files**: Detailed execution logs
3. **Cached Audio**: Generated TTS audio files (if caching enabled)
4. **Partial Results**: Batch-wise results during execution

### Result Structure

```json
{
  "ssl_model_config": "path/to/config.yml",
  "ssl_model_checkpoint": "model_latest.pt",
  "tts_model": "F5TTS",
  "attack_strategy": "voice_cloning_ssl",
  "total_trials": 1000,
  "overall_success_rate": 75.5,
  "same_speaker_success_rate": 82.3,
  "diff_speaker_success_rate": 68.7,
  "results": [
    {
      "label": 1,
      "success": true,
      "original_similarity": 0.85,
      "fake_similarity": 0.87,
      "target_transcript": "Hello world",
      "speaker_id": "id10001",
      "video_id": "video001",
      "audio_filename": "00001.wav"
    }
    // ... more results
  ]
}
```

## Studying Linguistic Sensitivity

This script enables studying how SSL models respond to linguistic content by:

1. **Voice Cloning Attacks**: Using TTS to generate fake audio with target speaker's voice
2. **Similarity Comparison**: Measuring how SSL embeddings change with cloned speech
3. **Linguistic Content**: Analyzing how different text content affects SSL representations
4. **Cross-Model Analysis**: Comparing results across different SSL architectures

## 🔍 Threshold Analysis Tools

### Find Optimal Threshold
Use the included threshold analysis script to understand your SSL model's decision boundaries:

```bash
# Find EER threshold for your SSL model
python find_ssl_threshold.py --config "models/your_ssl_model/config.yml"

# Skip plotting (for headless servers)
python find_ssl_threshold.py --config "models/your_ssl_model/config.yml" --no_plot
```

### Example Threshold Analysis
```bash
# Example usage
python example_find_threshold.py
```

This will show you:
- **EER Threshold**: The optimal balance point for your model
- **Operating Points**: Different security levels (high/medium/low security)
- **Score Distributions**: How same/different speaker scores are distributed
- **Visualizations**: DET curves, ROC curves, and score histograms

### Understanding Thresholds

| Threshold | Security | Usability | Use Case |
|-----------|----------|-----------|----------|
| 0.85+ | High | Low | Banking, Security |
| 0.68 (EER) | Balanced | Balanced | Research, General |
| 0.50-0.65 | Medium | High | Voice Assistants |

## Tips for Research

1. **Compare Multiple SSL Models**: Run attacks on different SSL architectures (SimCLR, DINO, etc.)
2. **Vary Text Content**: Use different transcripts to study linguistic sensitivity
3. **Analyze Embeddings**: Extract and analyze SSL embeddings for linguistic patterns
4. **Statistical Analysis**: Compare success rates across different linguistic features
5. **Ablation Studies**: Test with/without seeding, different TTS models, etc.
6. **Threshold Analysis**: Use threshold tools to understand each model's decision boundaries

## Troubleshooting

1. **SSL Model Loading**: Ensure config file and checkpoint paths are correct
2. **Protocol File Not Found**: 
   - Check your SSL model's `dataset.base_path` in the config file
   - Place protocol files in the base_path directory
   - Use relative filenames (e.g., `"voxceleb1_test_O"` not full paths)
   - The script will list all attempted locations if files aren't found
3. **Transcript File Not Found**: Follow same pattern as protocol files
4. **Memory Issues**: Reduce batch size if running out of memory
5. **Dataset Paths**: Verify VoxCeleb dataset paths are correct
6. **TTS Generation**: Check TTS model dependencies and voice cloning setup
7. **CUDA Issues**: Ensure proper CUDA setup for SSL models

### Common Path Issues

**Error**: `No such file or directory: 'data/data-list/vox1-O.txt'`
**Solution**: Use `--protocol_file "voxceleb1_test_O"` instead of full paths

**Error**: `Could not find protocol file`
**Solution**: Check that files are in your SSL model's `base_path` directory

## Citation

If you use this script in your research, please cite both the original attack methodology and the SSL framework used. 