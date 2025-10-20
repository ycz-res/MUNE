#!/usr/bin/env python3
"""
Demo: Generate synthetic data and call MUThresholdVisualizer to produce 4 figures
"""

import os
import numpy as np
from visualization import MUThresholdVisualizer


def main():
    save_dir = os.path.join('plot', 'training_curves_demo')
    os.makedirs(save_dir, exist_ok=True)

    vis = MUThresholdVisualizer(save_dir)

    # -------- 1) Simulate training history --------
    num_epochs = 12
    epochs = list(range(1, num_epochs + 1))
    # Smooth decreasing train/val losses
    train_losses = np.linspace(1.6, 0.95, num_epochs) + np.random.normal(0, 0.02, num_epochs)
    val_losses = np.linspace(1.5, 1.05, num_epochs) + np.random.normal(0, 0.02, num_epochs)

    # Simulate val metrics trending up
    precisions = np.linspace(0.45, 0.62, num_epochs) + np.random.normal(0, 0.01, num_epochs)
    recalls = np.linspace(0.97, 0.985, num_epochs) + np.random.normal(0, 0.002, num_epochs)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    ious = f1s / (2 - f1s + 1e-8)
    scores = 0.6 * f1s + 0.4 * ious

    for i, ep in enumerate(epochs):
        metrics = {
            'Precision': float(np.clip(precisions[i], 0, 1)),
            'Recall': float(np.clip(recalls[i], 0, 1)),
            'F1': float(np.clip(f1s[i], 0, 1)),
            'IoU': float(np.clip(ious[i], 0, 1)),
            'Score': float(np.clip(scores[i], 0, 1)),
        }
        vis.update_epoch(ep, float(train_losses[i]), float(val_losses[i]), metrics=metrics)

    # -------- 2) Simulate test results --------
    test_loss = float(val_losses[-1] + 0.02)
    test_metrics = {
        'Precision': float(np.clip(precisions[-1] - 0.01, 0, 1)),
        'Recall': float(np.clip(recalls[-1] - 0.002, 0, 1)),
        'F1': float(np.clip(f1s[-1] - 0.005, 0, 1)),
        'IoU': float(np.clip(ious[-1] - 0.005, 0, 1)),
        'Score': float(np.clip(scores[-1] - 0.005, 0, 1)),
    }
    vis.set_test_results(test_loss, test_metrics)

    # -------- 3) Simulate 20 test samples (CMAP + thresholds) --------
    num_samples = 20
    num_points = 500
    indices = list(range(10000, 10000 + num_samples))

    # CMAP normalized values [0,1], smooth-ish with noise
    x = np.linspace(0, 1, num_points)
    cmap = []
    for _ in range(num_samples):
        base = 0.5 + 0.4 * np.sin(2 * np.pi * (x + np.random.rand() * 0.2))
        noise = np.random.normal(0, 0.05, num_points)
        y = np.clip(base + noise, 0, 1)
        cmap.append(y)
    cmap = np.array(cmap, dtype=np.float32)

    # True thresholds: pick K random positions per sample and set to 1
    thresholds_true = np.zeros((num_samples, num_points), dtype=np.float32)
    mus_true = np.zeros((num_samples,), dtype=np.int32)
    rng = np.random.default_rng(123)
    for i in range(num_samples):
        k = int(rng.integers(3, 12))  # true MU count between 3 and 11
        mus_true[i] = k
        pos = rng.choice(num_points, size=k, replace=False)
        thresholds_true[i, pos] = 1.0

    # Pred thresholds: start from true, flip a few positions to simulate errors
    thresholds_pred = thresholds_true.copy()
    for i in range(num_samples):
        # remove some trues
        remove_k = int(max(1, mus_true[i] * 0.2))
        if mus_true[i] > 0:
            true_pos = np.where(thresholds_pred[i] == 1.0)[0]
            if len(true_pos) > 0:
                to_remove = rng.choice(true_pos, size=min(remove_k, len(true_pos)), replace=False)
                thresholds_pred[i, to_remove] = 0.0
        # add some false positives
        add_k = remove_k
        zero_pos = np.where(thresholds_pred[i] == 0.0)[0]
        to_add = rng.choice(zero_pos, size=min(add_k, len(zero_pos)), replace=False)
        thresholds_pred[i, to_add] = 1.0

    vis.set_sample_data(indices, cmap, thresholds_true, thresholds_pred, mus_true)

    # -------- 4) Generate figures --------
    vis.generate_four_figs()

    print(f"✅ Demo figures saved to: {save_dir}")


if __name__ == '__main__':
    main()


