# Brain Tumor RL — pacchetto refattorizzato

Refactoring dello script monolitico originale in moduli separati, con
aggiunta di: render umano/rgb_array dell'ambiente, checkpoint completi
(modello + VecNormalize insieme) e un valutatore visivo passo-passo.

## Struttura

```
config.py            costanti (azioni, reward shaping, soglie, curriculum)
utils.py              funzioni pure (IoU, distanza, coord planes, scheduler)
environment.py        BrainTumorRL_Env (Gymnasium) + render("human"/"rgb_array")
callbacks.py          VisionMetricsCallback (GradCAM), curriculum callbacks,
                       ModelCheckpointCallback (modello + VecNormalize insieme)
visual_evaluator.py   VisualEvaluator: rollout, video/GIF, griglie di step,
                       valutazione quantitativa + grafici aggregati
train.py              script principale (training / eval-only / watch)
dataset.py            NON incluso: è il modulo già esistente nel progetto
                       originale (get_datasets); va copiato accanto agli
                       altri file.
```

## Requisiti

```
pip install gymnasium stable-baselines3 sb3-contrib opencv-python torch numpy matplotlib imageio
```

`imageio` serve solo se si vogliono salvare GIF invece di MP4.

## Uso

### 1. Training completo (con checkpoint automatici e valutazione finale)

```
python train.py --total-timesteps 100000 --dataset-source kaggle \
                 --checkpoint-every 10000 --n-iterations 110
```

Ogni `--checkpoint-every` step viene salvata una coppia
`ppo_brain_tumor_<step>.zip` + `ppo_brain_tumor_<step>_vecnormalize.pkl`
in `./ppo_brain_tumor_logs/checkpoints/`. A differenza dello script
originale, le statistiche di normalizzazione vengono salvate ad OGNI
checkpoint, non solo a fine training: se il training si interrompe
prima, il checkpoint resta comunque utilizzabile.

Al termine, `evaluator.evaluate_dataset()` gira automaticamente sul
test set e produce:
- `test_eval/metrics_per_sample.csv` e `summary.txt` (come nello script originale)
- `test_eval/boxes/`, `test_eval/gradcam/`, `test_eval/failure_cases/`
- `test_eval/plots/` — istogramma IoU, IoU per bucket dimensione/contrasto,
  curva media IoU/reward nel tempo, pie del success rate
- `test_eval/episode_videos/` — video MP4 dei 5 casi peggiori, frame per frame

### 2. Solo valutazione visiva di un checkpoint già allenato

```
python train.py --eval-only \
  --model-path ./ppo_brain_tumor_logs/checkpoints/ppo_brain_tumor_500000.zip \
  --vecnorm-path ./ppo_brain_tumor_logs/checkpoints/ppo_brain_tumor_500000_vecnormalize.pkl
```

### 3. Osservare l'agente lavorare dal vivo su un campione del test set

Richiede un display disponibile (finestra OpenCV):

```
python train.py --watch-idx 3 \
  --model-path ...zip --vecnorm-path ..._vecnormalize.pkl
```

Questo apre una finestra `cv2.imshow` che mostra, passo dopo passo:
box verde = ground truth, box rosso = predizione dell'agente, e un HUD
con step corrente, azione scelta, IoU e reward istantanei. Salva anche
il video dell'episodio e una griglia riassuntiva degli step chiave.

### Uso programmatico del valutatore visivo

```python
from visual_evaluator import VisualEvaluator

evaluator = VisualEvaluator(model=model, test_ds=test_ds, output_dir="./out")
evaluator.save_episode_video(idx=12)        # video mp4 passo-passo
evaluator.save_episode_steps_grid(idx=12)   # griglia di frame chiave
evaluator.evaluate_dataset()                # metriche + grafici su tutto il test set
```

## Note sulle differenze rispetto all'originale

- La logica di reward/ambiente è invariata: nessuna modifica al comportamento
  di training già validato.
- `render_mode="human"` in `BrainTumorRL_Env` funziona solo se è disponibile
  un display X (in ambienti headless usare `render_mode="rgb_array"` +
  `VisualEvaluator.save_episode_video`, che non richiede display).
- `ModelCheckpointCallback` sostituisce il `CheckpointCallback` di SB3
  usato nello script originale: fa la stessa cosa ma include anche
  `vec_env.save()` ad ogni checkpoint.
