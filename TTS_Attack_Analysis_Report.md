# TTS Attack Analysis: Cross-Threshold EER Re-calculation and Combined Dataset Evaluation

## Executive Summary

This report describes a comprehensive analysis of Text-to-Speech (TTS) attacks on Self-Supervised Learning (SSL) speaker verification models. We developed a novel cross-threshold evaluation methodology that re-calculates Equal Error Rate (EER) using fixed thresholds optimized for different scenarios, providing insights into model robustness and optimal operating points.

## Methodology

### 1. Traditional EER Calculation vs. Fixed-Threshold Re-calculation

#### Traditional EER Approach
```
EER = Find threshold where FAR = FRR
Threshold = argmin|FAR(t) - FRR(t)|
```

#### Our Fixed-Threshold Re-calculation Approach
```python
def calculate_performance_at_threshold(scores, targets, threshold):
    predictions = (scores >= threshold).astype(int)
    
    # Confusion Matrix
    true_positives = sum((targets == 1) & (predictions == 1))
    false_positives = sum((targets == 0) & (predictions == 1))
    true_negatives = sum((targets == 0) & (predictions == 0))
    false_negatives = sum((targets == 1) & (predictions == 0))
    
    # Performance Metrics
    FAR = false_positives / sum(targets == 0)  # False Accept Rate
    FRR = false_negatives / sum(targets == 1)  # False Reject Rate
    error_rate = (false_positives + false_negatives) / len(targets)
    accuracy = (true_positives + true_negatives) / len(targets)
    
    return {
        'far': FAR * 100,
        'frr': FRR * 100, 
        'error_rate': error_rate * 100,
        'accuracy': accuracy * 100
    }
```

### 2. Cross-Threshold Analysis Framework

Our analysis evaluates three distinct threshold optimization scenarios:

1. **Original-Optimized Threshold**: EER threshold computed on real audio pairs only
2. **TTS-Optimized Threshold**: EER threshold computed on real vs. fake audio pairs only  
3. **Mixed-Optimized Threshold**: EER threshold computed on combined real and fake audio dataset

For each threshold, we re-evaluate performance on the original real audio test set to understand robustness.

### 3. Combined Dataset Creation

```python
def create_combined_trials():
    # Combine original VoxCeleb1 trials (real vs real)
    # + TTS attack trials (real vs fake)
    # Total: 75,222 trials (37,611 + 37,611)
    
    combined_trials = []
    combined_trials.extend(original_trials)  # Real vs Real
    combined_trials.extend(fake_trials)      # Real vs Fake
    
    return combined_trials
```

## Results

### Threshold Values Discovered

| Optimization Type | Threshold Value | Dataset Used | Purpose |
|------------------|----------------|--------------|---------|
| **Original-Optimized** | 0.294253 | Real vs Real | Baseline performance |
| **TTS-Optimized** | 0.248559 | Real vs Fake | Attack-specific threshold |
| **Mixed-Optimized** | 0.268705 | Real+Fake Combined | Balanced detection |

### Performance Analysis on Original Real Audio

| Threshold Type | Error Rate | FAR | FRR | Performance Impact |
|----------------|------------|-----|-----|-------------------|
| **Original-Optimized** | **2.56%** | - | - | ✅ Optimal for real audio |
| **Mixed-Optimized** | **2.88%** | 3.94% | 1.81% | 📊 +0.32% degradation |
| **TTS-Optimized** | **3.20%** | 5.16% | 1.24% | ⚠️ +0.64% degradation |

### TTS Attack Performance Analysis

| Threshold Type | Error Rate | FAR | FRR | Attack Effectiveness |
|----------------|------------|-----|-----|---------------------|
| **Original-Optimized** | **4.39%** | 1.81% | 6.98% | More conservative detection |
| **TTS-Optimized** | **3.64%** | - | - | ✅ Optimal for attack |

## Key Findings

### 1. Threshold Sensitivity Analysis

**Original Data Sensitivity to Threshold Changes:**
- Moving from optimal (0.294253) to TTS-optimized (0.248559): **+0.64% error increase**
- Moving from optimal (0.294253) to mixed-optimized (0.268705): **+0.32% error increase**

**Interpretation:** Lower thresholds make the system more lenient (higher FAR, lower FRR), while higher thresholds make it more conservative.

### 2. Attack Robustness Analysis

**TTS Attack Performance Across Thresholds:**
- At original threshold: 4.39% error rate
- At TTS-optimized threshold: 3.64% error rate
- **Robustness**: Only 0.75% difference, indicating attacks are robust across thresholds

### 3. Combined Dataset Insights

**Mixed-Optimized Threshold Benefits:**
- **Balanced Detection**: FAR (3.31%) ≈ FRR (3.32%)
- **Compromise Solution**: 0.32% performance cost on real data for significantly better fake detection
- **Real-World Applicability**: Suitable for environments with mixed real/fake audio

## Technical Implementation Details

### EER Threshold Computation
```python
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

def compute_eer_threshold(targets, scores):
    fpr, tpr, thresholds = roc_curve(targets, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_threshold = interp1d(fpr, thresholds)(eer)
    return float(eer_threshold)
```

### Embedding Caching System
- **Purpose**: Avoid redundant computation across multiple evaluations
- **Storage**: Pickled embeddings with unique cache keys per model
- **Performance**: Significantly reduced evaluation time for repeated runs

### Trial File Structure
```
Format: <label> <reference_audio> <target_audio>
- Label: 0 (different speaker) or 1 (same speaker)  
- Reference: Original real audio
- Target: Real audio (original) or fake audio (TTS-generated)
```

## Strategic Implications

### 1. Threshold Selection Strategy

**For Security-Critical Applications:**
- Use **Mixed-Optimized Threshold** (0.268705)
- Accepts 0.32% performance degradation on real audio
- Provides balanced protection against TTS attacks

**For Real-Audio-Optimized Applications:**
- Use **Original-Optimized Threshold** (0.294253)
- Best performance on legitimate audio
- May be vulnerable to sophisticated TTS attacks

### 2. Model Robustness Assessment

**SSL Model Vulnerabilities:**
- 42.1% increase in error rate under TTS attacks (2.56% → 3.64%)
- Threshold optimization provides limited attack improvement (0.75%)
- Model shows reasonable but not exceptional robustness

### 3. Real-World Deployment Considerations

**Threshold Adaptation:**
- Systems can dynamically adjust thresholds based on threat intelligence
- Mixed-optimized threshold provides good baseline for unknown environments
- Further optimization possible with domain-specific attack data

## Conclusions

1. **Cross-threshold analysis** reveals important trade-offs between real audio performance and attack resistance
2. **Mixed-optimized thresholds** offer practical compromise solutions for real-world deployment
3. **TTS attacks are moderately effective** against SSL models but not devastating
4. **Threshold selection** should be informed by expected attack prevalence and security requirements

## Future Work

- **Adaptive Thresholding**: Dynamic threshold adjustment based on confidence scores
- **Multi-Model Ensemble**: Combining multiple SSL models with different threshold strategies  
- **Attack-Aware Training**: Incorporating TTS-generated audio in SSL model training
- **Temporal Analysis**: Studying threshold effectiveness over time as TTS quality improves

---

**Technical Notes:**
- Analysis performed on VoxCeleb1 test set (37,611 trials)
- TTS generation using F5TTS with deterministic seeding
- SSL model: SimCLR E-ECAPA with SSPS k-means clustering
- All embeddings cached for reproducibility and performance 