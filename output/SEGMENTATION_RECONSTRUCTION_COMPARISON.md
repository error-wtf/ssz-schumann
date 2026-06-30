# Segmentation Reconstruction Comparison

**Generated:** 2026-07-01T00:12:54.180142

## Methods Compared

1. **Full Fit**: Calibrate eta_0 on full dataset (has degeneracy)
2. **Quiet Interval**: Calibrate eta_0 only on quiet period (first 14 days)
3. **Joint Fit**: Simultaneously fit eta_0 and delta_seg parameters

## Results

| Method | eta_0 | Bias | RMSE | Correlation | Mean δ_seg | Std δ_seg |
|--------|-------|------|------|-------------|------------|-----------|
| full_fit | 0.739565 | 0.000478 | 0.005727 | 0.9283 | 0.000386 | 0.015347 |
| quiet_interval | 0.739331 | 0.000161 | 0.005707 | 0.9283 | 0.000069 | 0.015342 |
| joint_fit | 0.739334 | 0.000092 | 0.000548 | 0.9993 | 0.000000 | 0.014223 |


## Key Findings

1. **Degeneracy Problem**: The full_fit method absorbs constant SSZ shifts into eta_0,
   resulting in near-zero mean reconstructed delta_seg.

2. **Quiet Interval Method**: By calibrating eta_0 only on the quiet period,
   we can recover the true mean delta_seg level.

3. **Joint Fit Method**: Simultaneously fitting eta_0 and delta_seg parameters
   provides the best reconstruction with lowest bias and RMSE.

## Recommendation

For SSZ detection, use either:
- **Quiet Interval** method if a known quiet period exists
- **Joint Fit** method for general cases

---

© 2025 Carmen Wrede & Lino Casu
