| approach | regime | model | postprocessing | roc_auc | f1 | sensitivity | specificity | average_precision | comment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline flow_only | all_epochs | XGBoost | none | 0.6007 | 0.3340 | 0.4767 | 0.6632 | 0.2737 | Single Flow modality baseline. |
| baseline respiratory_spo2_fusion | all_epochs | XGBoost | none | 0.6201 | 0.3419 | 0.5014 | 0.6618 | 0.3046 | Baseline respiratory + SpO2 fusion. |
| improved spo2_only | sleep_only | XGBoost | enhanced epoch context | 0.6725 | 0.4152 | 0.5264 | 0.7176 | 0.3717 | Best pre-temporal single-modality result. |
| improved respiratory_spo2_fusion | all_epochs | XGBoost | enhanced epoch context | 0.6657 | 0.3977 | 0.5486 | 0.7010 | 0.3506 | Best improved classical multimodal fusion. |
| improved late_fusion | all_epochs | XGBoost | late fusion | 0.6687 | 0.3542 | 0.4351 | 0.7688 | 0.3607 | Late fusion of modality-specific models. |
| temporal ensemble raw | sleep_only | MeanEnsemble | none | 0.6841 | 0.4255 | 0.5515 | 0.7124 | 0.3973 | SpO2 XGBoost + Flow/SpO2 HistGradientBoosting. |
| temporal ensemble causal smoothing | sleep_only | MeanEnsemble | rolling mean causal | 0.7014 | 0.4360 | 0.5568 | 0.7326 | 0.4243 | Label-free temporal smoothing without future probabilities. |
| temporal ensemble centered smoothing | sleep_only | MeanEnsemble | rolling mean centered | 0.7066 | 0.4349 | 0.5570 | 0.7304 | 0.4355 | Offline retrospective smoothing; uses future probabilities. |
| Simple 1D-CNN raw signals | sleep_only | 1D-CNN | raw probabilities | 0.5953 | 0.3453 | 0.4400 | 0.6951 | 0.2934 | Controlled raw-signal CNN baseline on Flow, SpO2, ribcage, and abdo. |
| ResNet1D 150s context | sleep_only | ResNet1D | offline 150s context | 0.5998 | 0.3756 | 0.5067 | 0.6669 | 0.2932 | Residual CNN with 150-second offline context. |
| DL improvement causal smoothing | sleep_only | CNN/ResNet1D | causal smoothing / mean probabilities | 0.6213 | 0.4018 | 0.4898 | 0.7236 | 0.3456 | Best DL-only post-processing summary from fixed OOF predictions. |
| Temporal ML ensemble | sleep_only | MeanEnsemble | temporal smoothing | 0.7066 | 0.4349 | 0.5570 | 0.7304 | 0.4355 | Best final ML result; offline centered temporal ensemble. |
