# Results Summary (Model Zoo)

This repo uses a frozen comparison protocol for fair experiments:

- Protocol: `experiments/model_zoo/protocol_10k_1k` (10k train / 1k val)
- Evaluation output: `experiments/results/<model_id>.json` + `experiments/predictions/<model_id>_val_preds.csv`
- Threshold optimization: stored inside each results JSON and mirrored in `config/thresholds.json` for the served model.

---

## Best currently served model

From `config/thresholds.json`:
- `model_version`: **distilmbert_lora_10k_v1**
- `global_threshold`: **0.15**

Metrics (optimized threshold on val):
- Source: `experiments/results/distilmbert_lora_10k_v1.json`
- Opt threshold: 0.15
- Opt F1: 0.4518
- Opt Precision / Recall: 0.4338 / 0.4713

---

## Comparable reference runs (same protocol)

### RuBERT + LoRA (10k/1k)
- Source: `experiments/results/rubert_base_lora_10k_v1.json`
- Opt threshold: 0.19
- Opt F1: 0.3702

### RuBERT title-only ablation (10k/1k)
- Source: `experiments/results/rubert_snippet_ablation_lora_10k_v1.json`
- Opt threshold: 0.09
- Opt F1: 0.2305


