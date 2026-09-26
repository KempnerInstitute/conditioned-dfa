# Integrated nDFA decision

Error factor: **short appendix note; no further error-factor rescue round**.

Practical gates: {'activity': True, 'early_activity': True, 'foof_dfa': True}. Covariance: True. DFA-specific timing: False. Temporary-activity efficiency: True.

| Prespecified primary contrast | Mean | 95% paired t interval | Holm p |
|---|---:|---|---:|
| cifar_activity | 2.263 | [1.9576, 2.5684] | 6.851e-07 |
| cifar_early_activity | 2.756 | [2.4483, 3.0637] | 1.374e-07 |
| cifar_foof_dfa | 3.554 | [3.2396, 3.8684] | 1.955e-08 |
| FOOF_DFA_minus_BP_gain | -0.229 | [-1.1476, 0.68957] | 1 |
| geometry_full | 3.205 | [3.0057, 3.4043] | 8.881e-10 |
| geometry_centered | 3.126 | [2.8166, 3.4354] | 5.025e-08 |
| timing_DFA | 1.186 | [0.92649, 1.4455] | 4.066e-05 |
| timing_DFA_minus_BP | 0.055 | [-0.45522, 0.56522] | 1 |
| error_vs_activity_accuracy | 0.384 | [0.017804, 0.7502] | 0.3149 |
| error_vs_activity_loss | 0.0070625 | [0.0018937, 0.012231] | 0.1549 |
| error_vs_early_diagonal_accuracy | 0.413 | [0.024534, 0.80147] | 0.3149 |
| error_vs_early_diagonal_loss | 0.0086475 | [0.0020426, 0.015252] | 0.1591 |
| error_direction_above_threshold | 0.37726 | [0.37489, 0.37963] | 1.037e-18 |
| mnist_nuisance_accuracy | -0.755 | [-1.373, -0.13696] | 0.1979 |
| mnist_nuisance_loss | -0.030316 | [-0.047267, -0.013365] | 0.03775 |
| mnist_nuisance_minus_clean_gain | -0.32 | [-0.94055, 0.30055] | 1 |
| fashion_nuisance_accuracy | 0.548 | [0.14493, 0.95107] | 0.1549 |
| fashion_nuisance_loss | -0.015309 | [-0.022969, -0.0076482] | 0.02023 |
| fashion_nuisance_minus_clean_gain | 0.467 | [0.028338, 0.90566] | 0.3149 |
| standard_background_accuracy | 0.143 | [-0.26817, 0.55417] | 1 |
| standard_background_loss | 0.0033421 | [-0.0048562, 0.011541] | 1 |

Positive loss contrasts mean lower CE. Accuracy contrasts are percentage points; the directional contrast is dimensionless. Intervals are individual, while tests use the declared global Holm family.

| Study / cell / family | Test accuracy | Test CE | Mean learning work (s) | Estimated linear-algebra FLOPs |
|---|---:|---:|---:|---:|
| confirm_foof_cifar / cifar10 / activity | 67.92% | 0.9435 | 200.0 | 2.262e+14 |
| confirm_foof_cifar / cifar10 / bp_adamw | 66.88% | 0.9631 | 200.0 | 2.49e+14 |
| confirm_foof_cifar / cifar10 / bp_sgd | 65.70% | 0.9999 | 200.0 | 2.831e+14 |
| confirm_foof_cifar / cifar10 / dfa_adamw | 65.66% | 0.9913 | 200.0 | 2.337e+14 |
| confirm_foof_cifar / cifar10 / dfa_no_bn | 61.30% | 1.1444 | 200.0 | 3.383e+14 |
| confirm_foof_cifar / cifar10 / dfa_sgd | 65.11% | 1.0108 | 200.0 | 2.668e+14 |
| confirm_foof_cifar / cifar10 / early_activity | 68.41% | 0.9449 | 200.0 | 2.318e+14 |
| confirm_foof_cifar / cifar10 / fd_dfa | 67.01% | 0.9602 | 200.0 | 4.378e+14 |
| confirm_foof_cifar / cifar10 / foof_bp | 70.03% | 0.9537 | 200.0 | 1.477e+15 |
| confirm_foof_cifar / cifar10 / foof_dfa | 69.21% | 0.9540 | 200.0 | 1.469e+15 |
| confirm_foof_nuisance / nuisance / activity_bn | 41.72% | 1.6317 | 20.0 | 3.076e+11 |
| confirm_foof_nuisance / nuisance / activity_none | 34.39% | 2.0405 | 20.0 | 3.88e+11 |
| confirm_foof_nuisance / nuisance / bp_sgd_bn | 14.26% | 2.2351 | 20.0 | 2.966e+11 |
| confirm_foof_nuisance / nuisance / bp_sgd_none | 15.07% | 2.5119 | 20.0 | 4.6e+11 |
| confirm_foof_nuisance / nuisance / bp_tuned_bn | 15.19% | 2.2401 | 20.0 | 2.549e+11 |
| confirm_foof_nuisance / nuisance / bp_tuned_none | 15.24% | 2.5493 | 20.0 | 4.084e+11 |
| confirm_foof_nuisance / nuisance / dfa_sgd_bn | 19.70% | 2.2323 | 20.0 | 2.32e+11 |
| confirm_foof_nuisance / nuisance / dfa_sgd_none | 16.42% | 2.8564 | 20.0 | 3.553e+11 |
| confirm_foof_nuisance / nuisance / dfa_tuned_bn | 15.61% | 2.2513 | 20.0 | 2.009e+11 |
| confirm_foof_nuisance / nuisance / dfa_tuned_none | 15.58% | 2.8502 | 20.0 | 3.024e+11 |
| confirm_foof_nuisance / nuisance / foof_bp_bn | 19.67% | 2.1597 | 20.0 | 5.194e+11 |
| confirm_foof_nuisance / nuisance / foof_bp_none | 23.37% | 2.3998 | 20.0 | 7.06e+11 |
| confirm_foof_nuisance / nuisance / foof_dfa_bn | 28.94% | 2.1097 | 20.0 | 4.609e+11 |
| confirm_foof_nuisance / nuisance / foof_dfa_none | 35.75% | 2.2066 | 20.0 | 6.277e+11 |
| confirm_error_cifar / cifar10 / activity | 66.87% | 0.9777 | 200.0 | 2.266e+14 |
| confirm_error_cifar / cifar10 / early_diagonal | 66.84% | 0.9793 | 200.0 | 2.26e+14 |
| confirm_error_cifar / cifar10 / early_k | 67.25% | 0.9706 | 200.0 | 2.179e+14 |
| confirm_error_cifar / cifar10 / persistent_k | 66.47% | 0.9703 | 200.0 | 1.824e+14 |
| confirm_error_mnist / mnist_relu / activity | 97.54% | 0.0834 | 60.0 | 4.1e+12 |
| confirm_error_mnist / mnist_relu / early_diagonal | 97.53% | 0.0842 | 60.0 | 4.183e+12 |
| confirm_error_mnist / mnist_relu / early_k | 97.77% | 0.0764 | 60.0 | 4.044e+12 |
| confirm_error_mnist / mnist_relu / persistent_k | 97.73% | 0.0764 | 60.0 | 3.65e+12 |
| confirm_benchmarks / fashion_clean / activity | 84.99% | 0.4534 | 60.0 | 4.227e+12 |
| confirm_benchmarks / fashion_clean / bp_bn | 85.67% | 0.4263 | 60.0 | 3.952e+12 |
| confirm_benchmarks / fashion_clean / dfa_bn | 84.91% | 0.4466 | 60.0 | 3.698e+12 |
| confirm_benchmarks / fashion_clean / fd_dfa | 82.39% | 0.7129 | 60.0 | 7.696e+12 |
| confirm_benchmarks / fashion_clean / foof_bp | 85.08% | 0.4356 | 60.0 | 1.802e+13 |
| confirm_benchmarks / fashion_clean / foof_dfa | 84.44% | 0.4951 | 60.0 | 1.649e+13 |
| confirm_benchmarks / fashion_input_nuisance / activity | 72.02% | 0.8419 | 60.0 | 4.229e+12 |
| confirm_benchmarks / fashion_input_nuisance / bp_bn | 70.21% | 0.8660 | 60.0 | 3.893e+12 |
| confirm_benchmarks / fashion_input_nuisance / dfa_bn | 71.47% | 0.8266 | 60.0 | 3.678e+12 |
| confirm_benchmarks / fashion_input_nuisance / fd_dfa | 65.50% | 1.0977 | 60.0 | 7.644e+12 |
| confirm_benchmarks / fashion_input_nuisance / foof_bp | 71.30% | 0.8685 | 60.0 | 1.788e+13 |
| confirm_benchmarks / fashion_input_nuisance / foof_dfa | 70.87% | 0.9376 | 60.0 | 1.783e+13 |
| confirm_benchmarks / fashion_label_noise / activity | 81.85% | 0.6797 | 60.0 | 4.2e+12 |
| confirm_benchmarks / fashion_label_noise / bp_bn | 82.50% | 0.6704 | 60.0 | 3.861e+12 |
| confirm_benchmarks / fashion_label_noise / dfa_bn | 81.73% | 0.6872 | 60.0 | 3.422e+12 |
| confirm_benchmarks / fashion_label_noise / fd_dfa | 76.16% | 0.8341 | 60.0 | 7.68e+12 |
| confirm_benchmarks / fashion_label_noise / foof_bp | 82.17% | 0.6733 | 60.0 | 1.782e+13 |
| confirm_benchmarks / fashion_label_noise / foof_dfa | 75.82% | 0.8441 | 60.0 | 1.75e+13 |
| confirm_benchmarks / mnist_background_standard / activity | 73.99% | 0.8103 | 60.0 | 4.225e+12 |
| confirm_benchmarks / mnist_background_standard / bp_bn | 72.95% | 0.8462 | 60.0 | 3.958e+12 |
| confirm_benchmarks / mnist_background_standard / dfa_bn | 73.85% | 0.8137 | 60.0 | 3.665e+12 |
| confirm_benchmarks / mnist_background_standard / fd_dfa | 71.20% | 1.0307 | 60.0 | 7.566e+12 |
| confirm_benchmarks / mnist_background_standard / foof_bp | 72.80% | 0.8506 | 60.0 | 1.8e+13 |
| confirm_benchmarks / mnist_background_standard / foof_dfa | 71.49% | 0.9641 | 60.0 | 1.732e+13 |
| confirm_benchmarks / mnist_clean / activity | 94.37% | 0.1889 | 60.0 | 3.929e+12 |
| confirm_benchmarks / mnist_clean / bp_bn | 95.83% | 0.1465 | 60.0 | 3.888e+12 |
| confirm_benchmarks / mnist_clean / dfa_bn | 94.80% | 0.1732 | 60.0 | 3.689e+12 |
| confirm_benchmarks / mnist_clean / fd_dfa | 94.83% | 0.1907 | 60.0 | 7.685e+12 |
| confirm_benchmarks / mnist_clean / foof_bp | 95.96% | 0.1567 | 60.0 | 1.79e+13 |
| confirm_benchmarks / mnist_clean / foof_dfa | 95.33% | 0.1721 | 60.0 | 1.779e+13 |
| confirm_benchmarks / mnist_input_nuisance / activity | 78.31% | 0.7035 | 60.0 | 4.221e+12 |
| confirm_benchmarks / mnist_input_nuisance / bp_bn | 79.18% | 0.6638 | 60.0 | 3.948e+12 |
| confirm_benchmarks / mnist_input_nuisance / dfa_bn | 79.07% | 0.6731 | 60.0 | 3.691e+12 |
| confirm_benchmarks / mnist_input_nuisance / fd_dfa | 76.95% | 0.7892 | 60.0 | 7.614e+12 |
| confirm_benchmarks / mnist_input_nuisance / foof_bp | 81.49% | 0.7268 | 60.0 | 1.801e+13 |
| confirm_benchmarks / mnist_input_nuisance / foof_dfa | 79.01% | 0.7072 | 60.0 | 1.784e+13 |
| confirm_benchmarks / mnist_label_noise / activity | 89.41% | 0.4966 | 60.0 | 4.167e+12 |
| confirm_benchmarks / mnist_label_noise / bp_bn | 90.02% | 0.4864 | 60.0 | 3.914e+12 |
| confirm_benchmarks / mnist_label_noise / dfa_bn | 90.48% | 0.4837 | 60.0 | 3.497e+12 |
| confirm_benchmarks / mnist_label_noise / fd_dfa | 91.06% | 0.4354 | 60.0 | 7.069e+12 |
| confirm_benchmarks / mnist_label_noise / foof_bp | 90.50% | 0.4837 | 60.0 | 1.796e+13 |
| confirm_benchmarks / mnist_label_noise / foof_dfa | 79.02% | 0.9634 | 60.0 | 1.777e+13 |
| confirm_epochs / cifar10 / activity_bn | 67.42% | 0.9640 | 207.3 | 2.254e+14 |
| confirm_epochs / cifar10 / dfa_bn | 65.00% | 1.0110 | 126.0 | 1.326e+14 |
| confirm_epochs / cifar10 / early_bn | 67.28% | 0.9737 | 134.9 | 1.558e+14 |
| confirm_geometry / cifar10 / centered | 67.90% | 0.9577 | 627.9 | 1.745e+15 |
| confirm_geometry / cifar10 / diagonal | 64.64% | 1.0390 | 637.0 | 1.745e+15 |
| confirm_geometry / cifar10 / diagonal_covariance_plus_mean | 64.77% | 1.0410 | 635.0 | 1.745e+15 |
| confirm_geometry / cifar10 / full | 67.98% | 0.9569 | 630.2 | 1.745e+15 |
| confirm_geometry / cifar10 / isotropic_covariance_plus_mean | 64.70% | 1.0429 | 639.4 | 1.745e+15 |
| confirm_geometry / cifar10 / raw_matched | 64.78% | 1.0343 | 114.8 | 1.326e+14 |
| confirm_geometry / cifar10 / sample_bridge | 67.98% | 0.9573 | 202.9 | 2.254e+14 |
| confirm_timing / cifar10 / bp_0.0003_early | 67.55% | 0.9951 | 137.5 | 1.65e+14 |
| confirm_timing / cifar10 / bp_0.0003_late | 66.42% | 1.0756 | 125.6 | 1.65e+14 |
| confirm_timing / cifar10 / dfa_0.0003_early | 65.76% | 1.0023 | 125.4 | 1.558e+14 |
| confirm_timing / cifar10 / dfa_0.0003_late | 64.58% | 1.0543 | 125.4 | 1.558e+14 |

Estimated leading-order dense linear-algebra FLOPs, FMA=2: forward, credit, weight products, moment products, solves, inverse refresh and calibration. Excludes pointwise activations, BN reductions, optimizer arithmetic, data transforms, diagnostics and exports; not measured hardware FLOPs. FD estimates include its documented matrix products separately. No FLOP-matched claim from time matching.

Standard and constructed background benchmarks are labeled separately. Failure of a preregistered prediction remains visible. Positive scalar error conditioning is algebraically A under norm matching, not an independent replication. Venue selection requires scientific review of these results; no acceptance guarantee follows from a gate.
