# System Paper Outline — SIDQ-T2

**Title:** SIDQ-T2: Corruption-Aware Self-Supervised Fusion for Robust Arabic Speech Deepfake Detection

## 1. Abstract

System description for ArA-DF 2026 Track 2 (Acoustic Robustness).
Multi-branch ensemble combining XLS-R, WavLM, and raw-waveform specialist
with corruption curriculum training and optimized score fusion.

## 2. Introduction

- Arabic speech deepfake detection under degraded conditions
- Track 2 focus: codec compression, noise, re-recording, social-media effects
- Challenge: detectors trained on clean audio fail on corrupted test conditions

## 3. Task Description

- Binary classification: bonafide (1) vs spoofed (0)
- EER as primary metric (threshold-independent, lower is better)
- Higher submission scores indicate stronger bonafide confidence

## 4. Data

- Training: 22,500 labeled utterances
- Development: 21,000 labeled utterances
- Track 2 development-test: 14,193 unlabeled utterances
- Track 2 final-test: 127,746 unlabeled utterances
- 16 kHz mono FLAC in WebDataset TAR shards

## 5. SIDQ Architecture

### 5.1 Branch A: SIDQ-XLSR
- XLS-R 300M frontend with learned layer mixing
- AASIST spectro-temporal graph attention backend

### 5.2 Branch B: SIDQ-WavLM
- WavLM Base+ with attentive statistics pooling
- Complementary feature representations

### 5.3 Branch C: SIDQ-Raw
- Sinc filterbank raw-waveform encoder
- Channel-artifact specialist

### 5.4 Score Fusion
- Per-model score normalization
- Constrained weight optimization minimizing validation EER

## 6. Corruption Curriculum

- Four severity profiles: none, mild, robust, hard
- Staged training: stabilization → robustness → hardening → selection
- Families: codec, noise, reverb, telephony, re-recording, laundering, RawBoost

## 7. Training Procedure

- Progressive frontend unfreezing
- Multi-view consistency regularization (clean + corrupted)
- Balanced class sampling
- AdamW with cosine schedule and warmup
- Early stopping on corruption-conditioned validation EER

## 8. Validation Protocol

- Fixed stratified train/validation split
- Clean validation EER
- Per-corruption-family EER via deterministic corruption bank
- Worst-group EER as selection objective
- Bootstrap confidence intervals

## 9. Fusion Method

- Z-score normalization per model
- Constrained optimization: non-negative weights, sum-to-one
- Cross-validated weight estimation on labeled validation predictions
- Score direction verified before and after fusion

## 10. Results

| System | Clean EER | Corrupted EER | Worst-Group EER |
|--------|-----------|---------------|-----------------|
| [To be filled from actual experiments] | | | |

## 11. Ablation Studies

| Component | Effect on EER |
|-----------|---------------|
| [To be filled from actual experiments] | |

## 12. Robustness Analysis

- Per-corruption-family performance
- Score drift analysis across conditions
- Complementarity of model branches

## 13. AWS Stress-Testing Methodology

- Controlled spoof generation via Amazon Polly
- Corruption ladder application
- Score fragility ranking
- Augmentation curriculum feedback loop

## 14. Limitations

- No speaker-group validation possible without verified speaker labels
- Corruption simulation approximates but does not replicate real channel effects
- Fusion weights optimized on available validation set may not transfer perfectly

## 15. Ethical Considerations

- System designed for defensive detection, not generation of deceptive speech
- No real-person voice cloning performed
- Synthetic outputs clearly marked in all manifests

## 16. Reproducibility

- All code and configurations committed
- Deterministic seeds and split manifests
- Experiment registry with git commit hashes
- Pre-commit quality enforcement

## 17. Conclusion

Multi-branch corruption-aware ensemble with curriculum training for robust
Arabic speech deepfake detection under acoustic degradation.
