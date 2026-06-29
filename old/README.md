# Codice legacy

Questa cartella contiene tutto ciò che **non fa parte** del pipeline attuale (`scripts/run_full_pipeline.py`).

## Contenuto

| Path | Descrizione |
|------|-------------|
| `scripts/run_full_pipeline.py` | Pipeline originale: U-Net + RL refinement |
| `scripts/run_full_pipeline2.py` | U-Net + RL ROI Finder |
| `scripts/train_baseline.py` | Training U-Net standalone |
| `scripts/train_rl.py` | Training RL standalone |
| `src/models/` | U-Net e loss |
| `src/train/trainer.py` | Trainer supervisionato U-Net |
| `src/train/fast_trainer.py` | Trainer ottimizzato DirectML |
| `src/train/rl_trainer.py` | Trainer RL originale (MaskRefinement) |
| `src/rl/agent.py` | DQNAgent originale |
| `src/rl/environment.py` | MaskRefinementEnv |
| `src/rl/environment2.py` | ROIFinderEnv legacy |
| `src/data/cached_dataset.py` | Cache in-memory |
| `src/data/fast_dataset.py` | DataLoader DirectML |
| `src/utils/visualization.py` | Plot U-Net / RL legacy |
| `README_legacy.md` | README originale del repo |

Non è necessario per eseguire il pipeline attuale. Conservato per riferimento e confronto.
