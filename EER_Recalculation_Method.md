# EER Re-calculation Methodology Using Fixed Thresholds

## Core Concept

Traditional EER calculation finds the **optimal threshold** where False Accept Rate (FAR) equals False Reject Rate (FRR). Our approach **fixes the threshold** and calculates performance metrics at that specific operating point.

## Traditional EER vs. Fixed-Threshold Approach

### Traditional EER Calculation
```python
# Traditional: Find threshold where FAR = FRR
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def traditional_eer(targets, scores):
    fpr, tpr, thresholds = roc_curve(targets, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_threshold = interp1d(fpr, thresholds)(eer)
    return eer * 100, eer_threshold  # EER percentage and optimal threshold
```

### Our Fixed-Threshold Re-calculation
```python
# Our approach: Use fixed threshold, calculate resulting performance
def calculate_performance_at_threshold(scores, targets, fixed_threshold):
    # Apply fixed threshold to get binary predictions
    predictions = (scores >= fixed_threshold).astype(int)
    
    # Build confusion matrix
    true_positives = sum((targets == 1) & (predictions == 1))   # Correct same-speaker
    false_positives = sum((targets == 0) & (predictions == 1))  # Wrong same-speaker  
    true_negatives = sum((targets == 0) & (predictions == 0))   # Correct different-speaker
    false_negatives = sum((targets == 1) & (predictions == 0))  # Wrong different-speaker
    
    # Calculate error rates
    total_same_speaker = sum(targets == 1)
    total_different_speaker = sum(targets == 0)
    
    FAR = false_positives / total_different_speaker    # False Accept Rate
    FRR = false_negatives / total_same_speaker         # False Reject Rate
    
    # Overall metrics
    total_errors = false_positives + false_negatives
    error_rate = total_errors / len(targets)           # Overall error rate
    accuracy = 1 - error_rate                          # Overall accuracy
    
    return {
        'threshold': fixed_threshold,
        'far': FAR * 100,                # False Accept Rate (%)
        'frr': FRR * 100,                # False Reject Rate (%)
        'error_rate': error_rate * 100,  # Total Error Rate (%)
        'accuracy': accuracy * 100       # Accuracy (%)
    }
```

## Step-by-Step Process

### Step 1: Obtain Threshold from Different Scenarios
We extract EER thresholds from three different optimization scenarios:

```python
# Scenario 1: Original real audio data
original_threshold = compute_eer_threshold(real_targets, real_scores)
# Result: 0.294253

# Scenario 2: TTS attack data  
tts_threshold = compute_eer_threshold(fake_targets, fake_scores)
# Result: 0.248559

# Scenario 3: Combined real + fake data
combined_threshold = compute_eer_threshold(combined_targets, combined_scores) 
# Result: 0.268705
```

### Step 2: Re-evaluate Original Data with Each Threshold
```python
# Take original real audio test set
original_scores = [...]  # Cosine similarity scores from SSL model
original_targets = [...]  # Ground truth labels (0 or 1)

# Re-calculate performance at each threshold
perf_at_original = calculate_performance_at_threshold(
    original_scores, original_targets, original_threshold)

perf_at_tts = calculate_performance_at_threshold(
    original_scores, original_targets, tts_threshold)

perf_at_mixed = calculate_performance_at_threshold(
    original_scores, original_targets, combined_threshold)
```

### Step 3: Compare Results
| Threshold Source | Threshold Value | Error Rate on Real Data | Interpretation |
|-----------------|----------------|------------------------|----------------|
| Original-optimized | 0.294253 | 2.56% | ✅ Best for real data |
| Mixed-optimized | 0.268705 | 2.88% | 📊 Balanced approach |
| TTS-optimized | 0.248559 | 3.20% | ⚠️ Optimized for attacks |

## Why This Matters

### 1. **Threshold Transferability**
- Shows how well a threshold optimized for one scenario works in another
- Critical for real-world deployment where data distribution may vary

### 2. **Robustness Analysis**
- Reveals model sensitivity to operating point changes
- Helps identify threshold ranges where performance degrades significantly

### 3. **Attack Impact Assessment**
- Quantifies how TTS attacks affect optimal operating points
- Measures performance cost of defending against attacks

## Mathematical Interpretation

### Threshold Effects on Decision Boundary
```
If threshold is HIGHER (more conservative):
├── FAR decreases (fewer false accepts)
├── FRR increases (more false rejects)  
└── System becomes more restrictive

If threshold is LOWER (more lenient):
├── FAR increases (more false accepts)
├── FRR decreases (fewer false rejects)
└── System becomes more permissive
```

### Our Results Interpretation
```
Original threshold (0.294253) → Conservative, optimized for real audio
TTS threshold (0.248559)      → Lenient, allows more TTS attacks through  
Mixed threshold (0.268705)    → Balanced, compromise between both
```

## Practical Implementation

### Complete Workflow
```python
class CrossThresholdEvaluator:
    def __init__(self, model):
        self.model = model
        self.thresholds = {}
        self.results = {}
    
    def compute_eer_threshold(self, scores, targets, name):
        """Compute and store EER threshold for a scenario"""
        fpr, tpr, thresholds = roc_curve(targets, scores, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        threshold = interp1d(fpr, thresholds)(eer)
        self.thresholds[name] = float(threshold)
        return self.thresholds[name]
    
    def evaluate_at_threshold(self, scores, targets, threshold_name, test_name):
        """Evaluate test data at a specific threshold"""
        threshold = self.thresholds[threshold_name]
        result = calculate_performance_at_threshold(scores, targets, threshold)
        self.results[f"{test_name}_at_{threshold_name}"] = result
        return result
    
    def cross_threshold_analysis(self):
        """Perform complete cross-threshold analysis"""
        for threshold_name in self.thresholds:
            for test_name, (scores, targets) in self.test_sets.items():
                self.evaluate_at_threshold(scores, targets, threshold_name, test_name)
```

## Key Insights from Our Analysis

1. **Threshold Hierarchy**: Original (0.294) > Mixed (0.269) > TTS (0.249)
2. **Performance Degradation**: Using suboptimal thresholds costs 0.32-0.64% error rate
3. **Attack Robustness**: TTS attacks work reasonably well across different thresholds
4. **Balanced Solution**: Mixed-optimized threshold offers good compromise for real-world deployment

This methodology provides a comprehensive framework for understanding model behavior across different operating conditions and threat scenarios. 