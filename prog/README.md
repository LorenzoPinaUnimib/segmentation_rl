# Localizzazione di Tumori Cerebrali tramite Reinforcement Learning

# Relazione di Progetto di Sistemi Complessi: Modelli e Simulazione di Pina Lorenzo 894396 e Rancati Simone 900052

# DA AGGIUNGERE spiegazione modello e reward, grafici andamento reward, spiegazione apprendimento agente RL, tipologie di modifiche che abbiamo provato e bibliografia

Abbiamo modificato più volte modello, reward, decadimento di epsilon, dimensione box iniziale del modello, spazio delle azioni (con o senza terminazione), modalità di acquisizione feature dall'immagine, tentato curriculum learning.

## Indice

1. [Introduzione e motivazione del problema](#1-introduzione-e-motivazione-del-problema)
2. [Panoramica architetturale del sistema](#2-panoramica-architetturale-del-sistema)
3. [Formalizzazione come processo decisionale di Markov](#3-formalizzazione-come-processo-decisionale-di-markov)
4. [Pipeline dei dati: acquisizione, ispezione, dataset](#4-pipeline-dei-dati-acquisizione-ispezione-dataset)
5. [Preprocessing e augmentation delle immagini](#5-preprocessing-e-augmentation-delle-immagini)
6. [L'ambiente `BrainTumorRL_Env`](#6-lambiente-braintumorrl_env)
7. [Reward shaping: anatomia dettagliata di ogni componente](#7-reward-shaping-anatomia-dettagliata-di-ogni-componente)
8. [Il ramo discreto: 9 azioni, action masking e MaskablePPO](#8-il-ramo-discreto-9-azioni-action-masking-e-maskableppo)
9. [Il ramo continuo: azioni `[dx,dy,dw,dh]` e SAC](#9-il-ramo-continuo-azioni-dxdydwdh-e-sac)
10. [Il warm-start supervisionato: `localizer.py`](#10-il-warm-start-supervisionato-localizerpy)
11. [Curriculum learning: dalla rampa fissa all'auto-paced](#11-curriculum-learning-dalla-rampa-fissa-allauto-paced)
12. [Callback di monitoraggio e spiegabilità (GradCAM)](#12-callback-di-monitoraggio-e-spiegabilità-gradcam)
13. [Checkpointing e gestione dello stato di training](#13-checkpointing-e-gestione-dello-stato-di-training)
14. [Valutazione quantitativa e visiva: `visual_evaluator.py`](#14-valutazione-quantitativa-e-visiva-visual_evaluatorpy)
15. [Il flusso end-to-end di `train.py`](#15-il-flusso-end-to-end-di-trainpy)
16. [Interfaccia a riga di comando e casi d'uso](#16-interfaccia-a-riga-di-comando-e-casi-duso)
17. [Cronologia evolutiva del progetto (FIX v2 → v7)](#17-cronologia-evolutiva-del-progetto-fix-v2--v7)
18. [Analisi critica, limiti e lavori futuri](#18-analisi-critica-limiti-e-lavori-futuri)
19. [Conclusioni](#19-conclusioni)
20. [Bibliografia](#20-bibliografia)

---

## 1. Introduzione e motivazione del problema

Il progetto affronta il compito di localizzare un tumore cerebrale all'interno di un'immagine di risonanza magnetica (MRI) per mezzo di Reinforcement Learning: un agente osserva l'immagine e la posizione corrente di un riquadro (bounding box), e ad ogni passo temporale decide come modificarlo (spostandolo, ridimensionandolo o fermandosi) fino a farlo coincidere il più possibile con l'area occupata dal tumore.

Questa impostazione possiede alcune proprietà interessanti rispetto alla segmentazione diretta:

- **Interpretabilità del processo**: ogni traiettoria dell'agente è una sequenza di decisioni ispezionabili (quali azioni, in che ordine, con quale guadagno di IoU).
- **Segnale di supervisione debole**: il segnale di reward deriva dalla maschera di verità (ground truth) solo indirettamente, tramite la metrica di Intersection-over-Union (IoU) tra il box corrente e il box che racchiude la maschera.
- **Naturale predisposizione al curriculum learning**: la difficoltà del compito (distanza iniziale del box dal target) è un parametro continuo e facilmente controllabile, aprendo la porta a strategie di apprendimento progressivo.

---

## 2. Panoramica architetturale del sistema (da revisionare)

Il pacchetto è organizzato in moduli con responsabilità nettamente separate, un buon esempio di *separation of concerns* applicato a un progetto di ricerca:

| Modulo | Responsabilità |
|---|---|
| `config.py` | Costanti globali condivise: definizione delle azioni, coefficienti del reward shaping, soglie di valutazione, durate di default del curriculum. |
| `dataset_inspector.py` | Euristiche di file-system scanning per scoprire automaticamente coppie (immagine, maschera) o annotazioni COCO in una cartella arbitraria. |
| `dataset.py` | Classi `Dataset` PyTorch (COCO, file-pair, sintetico), pipeline di preprocessing condivisa, split train/val/test, factory dei `DataLoader`. |
| `transforms.py` | Normalizzazione, white balance, CLAHE, denoising, augmentation via Albumentations (o fallback numpy puro). |
| `environment.py` | `BrainTumorRL_Env`: l'ambiente Gymnasium vero e proprio, con i due rami di step (discreto/continuo), reward shaping, rendering. |
| `utils.py` | Funzioni pure e stateless: IoU, distanza tra centri, costruzione del vettore box normalizzato, scheduler lineare, mask function per l'`ActionMasker`. |
| `localizer.py` | Regressore CNN supervisionato per il warm-start del box iniziale. |
| `callbacks.py` | Tutti i callback SB3: logging metriche/GradCAM, varianti di curriculum (a step fissi, a stage, adattivo), checkpointing combinato modello+normalizzazione. |
| `visual_evaluator.py` | Rollout ispezionabili (video, griglie), valutazione quantitativa aggregata con breakdown per bucket dimensione/contrasto, grafici Matplotlib. |
| `train.py` | Punto di ingresso: parsing CLI, costruzione dataset/ambienti vettorializzati, istanziazione algoritmo (MaskablePPO o SAC), ciclo di training, valutazione finale. |

Il flusso concettuale complessivo è il seguente:

```
dataset_inspector.py ──► dataset.py ──► get_datasets(cfg)
                                             │
                                             ▼
                              train_ds, val_ds, test_ds
                                             │
                     ┌───────────────────────┼─────────────────────────┐
                     ▼                       ▼                         ▼
           (opz.) localizer.py     environment.py (BrainTumorRL_Env)   visual_evaluator.py
           warm-start regressore    wrapped in SubprocVecEnv +               (fine training)
                     │              VecMonitor + VecNormalize
                     └───────────────────────┼─────────────────────────┘
                                             ▼
                                 MaskablePPO  oppure  SAC
                                 (sb3-contrib)   (stable-baselines3)
                                             │
                              callbacks.py: VisionMetricsCallback,
                              AdaptiveCurriculumCallback,
                              StopCurriculumCallback (solo discreto),
                              ModelCheckpointCallback, Eval*Callback
                                             │
                                             ▼
                                   train.py orchestration loop
                                   (n_iterations × model.learn())
```

---

## 3. Formalizzazione come processo decisionale di Markov

Per inquadrare rigorosamente il problema come RL, lo si definisce come un Processo Decisionale di Markov (MDP) parzialmente osservabile tramite immagine, nella forma a 5-tupla $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

### 3.1 Stato e osservazione

Lo stato ad ogni istante $t$ è composto da:
- l'immagine corrente $I \in \mathbb{R}^{C \times H \times W}$ (fissa per l'intero episodio, $C=3$, $H=W=224$);
- i parametri geometrici del box predetto $(c_x, c_y, w, h)$ (centro e dimensioni, in pixel);
- il box di verità $(gt_x, gt_y, gt_w, gt_h)$, derivato dal bounding box che racchiude la maschera binaria (usato solo per calcolare il reward).

L'osservazione fornita alla policy è uno spazio `gymnasium.spaces.Dict` con due chiavi (si veda `environment.py`, `_get_obs`):

```python
observation_space = spaces.Dict({
    "image":   spaces.Box(low=0, high=255, shape=(C+1, H, W), dtype=np.uint8),
    "box_vec": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
})
```

- `image`: l'immagine RGB (3 canali) concatenata con una maschera binaria rasterizzata del box corrente (1 canale aggiuntivo, disegnata con `cv2.rectangle(..., thickness=-1)`), per un totale di 4 canali.
- `box_vec`: il vettore $(c_x/W,\; c_y/H,\; w/W,\; h/H)$ normalizzato in $[0,1]$ (funzione `build_box_vec` in `utils.py`).

Questa è una scelta progettuale esplicitamente motivata nel codice (commento "FIX (Dict observation space)"): in una versione precedente le coordinate del box venivano "spalmate" su 4 piani immagine interi a valore costante (8 canali totali), un formato che (a) raddoppiava il numero di canali dell'immagine — e quindi la RAM del replay buffer SAC, poiché `DictReplayBuffer` non supporta `optimize_memory_usage` — e (b) costringeva la rete a dedurre il valore numerico del box da un piano spaziale costante invece di riceverlo come input diretto. Passare a un vettore numerico esplicito dimezza i canali immagine (da 8 a 4) e fornisce l'informazione geometrica in una forma più facile da apprendere.

Su questo tipo di osservazione la rete usata da Stable-Baselines3/`sb3-contrib` è un `CombinedExtractor` (attivato passando `policy="MultiInputPolicy"`), che instrada automaticamente ogni chiave verso il sotto-estrattore appropriato — una CNN (`NatureCNN`, Mnih et al., 2015) per `"image"` e un semplice flatten per `"box_vec"` — per poi concatenare le feature risultanti prima delle teste policy/valore.

### 3.2 Spazio delle azioni

Lo spaziod delle azioni è discreto (`spaces.Discrete(9)`): 4 direzioni di traslazione, 4 direzioni di ridimensionamento e 1 azione di stop.

### 3.3 Dinamica di transizione

La geometria del box viene aggiornata e poi vincolata (`clip`) affinché resti dentro l'immagine e con dimensioni minime di 12 pixel. L'unica fonte di stocasticità è nel `reset()`, che determina l'immagine campionata dal dataset e la posizione e dimensione iniziale del box.

### 3.4 Funzione di reward (da aggiornare una volta scelto cosa mettere)

Combinazione lineare pesata di più segnali (dettagliata in §7): guadagno di IoU, avvicinamento del centro, penalità di tempo, penalità di sovradimensionamento, penalità di oscillazione/jerk, e un bonus terminale calcolato alla fine dell'episodio.

### 3.5 Orizzonte ed episodio

L'episodio può terminare anticipatamente (`terminated=True`) se l'agente sceglie l'azione stop, altrimenti viene troncato (`truncated=True`) al raggiungimento di `max_steps`.

### 3.6 Fattore di sconto

$\gamma = 0.99$ bilancia l'orizzonte di pianificazione con la stabilità delle stime di valore.

---

## 4. Pipeline dei dati: acquisizione, ispezione, dataset (da rivedere)

### 4.1 `dataset_inspector.py`: scoperta automatica delle coppie immagine/maschera

Il modulo implementa una cascata di euristiche per adattarsi a diversi formati di dataset scaricati da fonti eterogenee (in particolare Kaggle), in ordine di priorità decrescente:

1. **Annotazioni COCO** (`find_coco_jsons` + `load_coco_pairs`): cerca ricorsivamente file `.json` il cui nome contiene "annotation" e verifica che contengano le chiavi `images`/`annotations`; costruisce coppie (percorso immagine, dizionario annotazioni con poligoni). Ha priorità massima perché le maschere COCO (poligoni vettoriali) sono più precise di maschere raster salvate separatamente.
2. **Cartelle divise per split** (`build_pairs_split_dirs`): cerca sottocartelle `train/val/validation/test`, ciascuna con sottocartelle parallele immagine/maschera.
3. **Cartelle parallele nella root** (`build_pairs_by_parallel_dirs`): cerca coppie di sottocartelle in cui una ha un nome che matcha la regex `MASK_KEYWORDS` (`mask|label|seg|annotation|gt|ground_truth|truth`, case-insensitive) e l'altra no, poi appaia i file per nome (stem).
4. **Matching per nome file** (`build_pairs_by_name`): fallback finale su tutta la struttura, normalizzando i nomi file (rimuovendo le keyword di maschera) e appaiando per uguaglianza di stringa.

Se nessuna strategia produce coppie, il sistema stampa un warning e ripiega su un **dataset sintetico** (§4.3), garantendo che il training non si blocchi mai per assenza di dati validi — una scelta di robustezza ingegneristica importante per un progetto sviluppato/testato iterativamente.

### 4.2 Le tre implementazioni di `Dataset`

Tutte e tre ereditano da una classe base condivisa `_BaseTumorDataset`, che fattorizza augmentation, preprocessing e packaging tensoriale in un unico punto (`_finalize`), eliminando la ridondanza/rischio di divergenza che si avrebbe reimplementando la stessa logica tre volte:

- **`COCOAnnotationDataset`**: rasterizza i poligoni COCO in maschere binarie al volo (`polygons_to_mask`, via `cv2.fillPoly`).
- **`BrainTumorDataset`**: carica coppie (immagine, file maschera) da disco.
- **`SyntheticBrainDataset`**: genera immagini "MRI-like" procedurali — sfondo sinusoidale + rumore gaussiano, "cranio" ellittico scuro, tumore ellittico ruotato casualmente con eventuali "satelliti" (macchie secondarie), **e polarità casuale** (tumore più chiaro o più scuro del background con probabilità 50/50) — usata sia come fallback quando l'ispezione fallisce sia per test rapidi senza dipendenza da dataset esterni.

### 4.3 Metadati derivati per campione

Ogni campione restituito include, oltre a `image` e `mask`:
- `area_ratio`: frazione dell'immagine occupata dalla maschera (usata per il bucket dimensionale in valutazione, §14);
- `polarity`: `"bright"` o `"dark"` a seconda che l'intensità media dentro la maschera sia maggiore o minore di quella del background (funzione `classify_tumor_polarity`).

Questi metadati sono calcolati **prima** dell'augmentation, sui dati grezzi, per restare stabili indipendentemente dalla vista aumentata effettivamente vista dall'agente in un dato step — altrimenti lo stratified sampling e le metriche di valutazione per bucket diventerebbero rumorose.

### 4.4 Split e configurazione

Lo split è `train_ratio=0.8`, `val_ratio=0.1` (il restante 0.1 è test), con seed fisso `42` per riproducibilità (`_split_pairs`, shuffle con `random.Random(seed)`). Il dataset di default (via CLI/`env`) è quello Kaggle `pkdarabi/brain-tumor-image-dataset-semantic-segmentation`, scaricato tramite `kagglehub`; in caso di fallimento del download il sistema ripiega automaticamente sul sintetico.

---

## 5. Preprocessing e augmentation delle immagini

La pipeline di preprocessing (`apply_preprocessing` in `dataset.py`, implementata in `transforms.py`) è applicata in un ordine preciso e motivato:

1. **White balance "Gray World"** (`white_balance_gray_world`): riscala i tre canali RGB in modo che le loro medie coincidano, correggendo dominanti di colore spurie (es. tinte introdotte da acquisizione via schermo, compressione JPEG, export da scanner diversi) che non portano segnale anatomico utile. Guadagni limitati a $[0.7, 1.4]$ per non alterare in modo estremo immagini quasi monocromatiche. Attivo di default.
2. **CLAHE** (Contrast Limited Adaptive Histogram Equalization, Zuiderveld, 1994) applicato sul canale di luminosità nello spazio colore LAB (non sui canali RGB direttamente, per non distorcere il bilanciamento colore appena corretto): aumenta il contrasto locale ai bordi tumore/tessuto, tipicamente poco marcati in MRI. Parametri: `clip_limit=2.0`, `tile_grid_size=(8,8)`. Attivo di default.
3. **Denoising leggero** (bilateral filter, Tomasi & Manduchi, 1998): disattivato di default, da attivare solo su dataset visibilmente rumorosi (rischia di ammorbidire tumori piccoli).
4. **Clipping robusto delle intensità** (percentili 1–99, opzionale): riduce l'influenza di outlier estremi prima della normalizzazione, con percentili scelti conservativamente per non tagliare via tumori piccoli e molto chiari/scuri che potrebbero cadere nella coda tagliata.
5. **Normalizzazione**: modalità `minmax` selezionata di default in `build_dataset_config` (`config.py` in `train.py`).
6. **Skull-stripping approssimato** (opzionale, disattivo di default): soglia di Otsu + chiusura/apertura morfologica + selezione della componente connessa più grande, usato solo come canale di contesto aggiuntivo, mai come crop distruttivo (un crop aggressivo rischierebbe di tagliare tumori vicini al bordo del cranio).

### 5.1 Augmentation (solo split di training)

Implementata via **Albumentations** (Buslaev et al., 2020), con fallback numpy puro (`FallbackTransform`) se la libreria non è disponibile — entrambi i percorsi rispettano la stessa configurazione, per non introdurre comportamenti divergenti a seconda dell'ambiente di esecuzione:

- **Horizontal flip** (`p=0.5`): valido in radiologia per scansioni assiali.
- **Vertical flip**: **disattivo di default** (`p=0.0`), perché un cervello in scansione assiale non ha simmetria alto/basso — capovolgerlo produce un'anatomia impossibile.
- **Rotazione** limitata a ±10° (`border_mode=REFLECT_101`): rotazioni ampie (>15–20°) sono irrealistiche per scansioni già allineate e introdurrebbero bordi neri interpolati sfruttabili come scorciatoia spuria dalla rete.
- **Brightness/contrast jitter** (`limit=0.2` per entrambi): leva principale per la robustezza al contrasto (tumori sia iper- che ipo-intensi); valori più aggressivi rischierebbero di rendere un tumore ipointenso isointenso al background, cancellando il segnale target.
- **Gamma jitter** (opzionale) e **rumore gaussiano** (`p=0.3`).

---

## 6. L'ambiente `BrainTumorRL_Env`

### 6.1 Interfaccia Gymnasium

`BrainTumorRL_Env` eredita da `gymnasium.Env` e implementa l'interfaccia standard `reset()`/`step()`/`render()`/`close()`, oltre a `action_masks()` (richiesto da `ActionMasker`/MaskablePPO nel ramo discreto). Supporta due `render_mode`:
- `"human"`: finestra `cv2.imshow` in tempo reale (richiede display);
- `"rgb_array"`: ritorna il frame BGR corrente come array numpy, per registrare video/GIF senza display (usato da `VisualEvaluator`).

Il frame renderizzato (`_render_frame_bgr`) disegna il box di verità in verde, il box predetto in rosso, e un HUD testuale semi-trasparente con step corrente, ultima azione, IoU istantaneo e reward istantaneo.

### 6.2 Logica di `reset()`: come nasce il box iniziale

Ad ogni reset viene campionata un'immagine casuale dal dataset e calcolato il box di verità $(gt_x, gt_y, gt_w, gt_h)$ come bounding box dei pixel della maschera con valore $>0.5$ (fallback a un box centrale $W/4 \times H/4$ se la maschera è vuota). Il box iniziale predetto viene poi generato in uno di due modi, a seconda che sia disponibile un **localizzatore supervisionato** (§10):

**Senza localizzatore** (comportamento "originale"): blend stocastico tra due modalità, controllato dal parametro di difficoltà $d \in [0,1]$ (`init_difficulty`):
- con probabilità $d$: box **completamente casuale** dentro l'immagine (`random_cx/cy` uniformi in $[0.2W, 0.8W]$, dimensioni in $[0.15, 0.40]$ delle dimensioni immagine);
- con probabilità $1-d$: box con **jitter attorno alla ground truth** ("easy"), con ampiezza del jitter crescente linearmente con $d$: `pos_jitter_frac = 0.10 + 0.50d`, `size_jitter_frac = 0.10 + 0.60d`.

**Con localizzatore** (`localizer_fn` fornita): il box iniziale parte dalla **predizione del regressore CNN** più rumore la cui scala dipende da $d$ e dalle dimensioni *predette* (non da quelle di ground truth, per evitare data leakage — l'informazione sulle dimensioni predette è disponibile anche a inferenza reale su immagini mai viste). In questo caso $d$ cambia semanticamente significato: non più "quanto lontano parto dalla GT" ma "quanto rumore aggiungo sopra la predizione del regressore".

In entrambi i casi il box viene infine vincolato (`clip`) a dimensioni $\geq 12$px e centro tale da restare interamente dentro l'immagine.

### 6.3 Setter usati dai callback di curriculum

L'ambiente espone tre metodi mutatori invocati dai callback (via `training_env.env_method(...)`, poiché gli ambienti girano in sotto-processi separati tramite `SubprocVecEnv`):

```python
def set_min_steps_before_stop(self, value): ...   # numero minimo di step prima che STOP sia permesso
def set_step_frac(self, frac): ...                # frazione della dimensione corrente del box usata come passo di movimento
def set_init_difficulty(self, value): ...          # difficoltà d del reset (clip in [0,1])
```

Questo pattern (stato mutabile dell'ambiente controllato esternamente da un callback SB3) è il meccanismo con cui l'intero curriculum learning del progetto è implementato — si veda §11.

---

## 7. Reward shaping: anatomia dettagliata di ogni componente

Tutte le costanti sono definite in `config.py`. Il reward è una somma di termini, ciascuno *clippato* indipendentemente per limitare la varianza complessiva del segnale (una pratica comune in reward engineering per stabilizzare l'apprendimento — Ng, Harada & Russell, 1999, sulla *potential-based reward shaping* come garanzia teorica di invarianza della policy ottima).

### 7.1 Termine di avanzamento IoU (`delta_iou`)

$$
r_{\Delta IoU} = \text{clip}\big((IoU_t - IoU_{t-1}) \cdot 25.0,\; -10,\; +10\big)
$$

Premia/punisce il **guadagno marginale** di IoU rispetto allo step precedente, non il valore assoluto — questo rende il segnale denso (disponibile ad ogni step, non solo a fine episodio) e indipendente dal punto di partenza dell'episodio.

### 7.2 Termine di avvicinamento del centro (`delta_dist`)

$$
r_{\Delta dist} = \text{clip}\big((d_{t-1} - d_t) \cdot 0.05,\; -1,\; +1\big)
$$

dove $d_t$ è la distanza euclidea tra i centri del box predetto e di verità (`compute_center_distance`). Fornisce un segnale utile anche quando l'IoU è ancora nulla (box e GT non si sovrappongono affatto, quindi $IoU_t = IoU_{t-1} = 0$ e il termine precedente sarebbe cieco) — un classico esempio di *reward shaping* per problemi con reward sparso.

### 7.3 Penalità di tempo (`TIME_PENALTY = -0.01`)

Costante ad ogni step, per incentivare soluzioni rapide ed evitare che l'agente indugi indefinitamente.

### 7.4 Penalità di sovradimensionamento (`_oversize_penalty`)

$$
r_{oversize} = -\max\Big(0,\; \frac{w \cdot h}{W \cdot H} - 0.5\Big) \cdot 4.0
$$

Scoraggia la strategia degenere "box grande quanto tutta l'immagine" (che darebbe comunque un IoU non nullo se il tumore è contenuto al suo interno, ma è una soluzione poco informativa).

### 7.5 Penalità di oscillazione (ramo discreto) / smoothness (ramo continuo)

- **Discreto**: `OSCILLATION_PENALTY = 0.01` applicato quando l'azione corrente è l'esatto opposto dell'azione precedente (dizionario `OPPOSITE_ACTIONS`, es. LEFT dopo RIGHT), per scoraggiare micro-oscillazioni improduttive.
- **Continuo**: `_smoothness_penalty`, equivalente concettuale ma calcolato come penalità quadratica sul "jerk" (variazione tra vettori di azione consecutivi):
$$
r_{smooth} = -\|\mathbf{a}_t - \mathbf{a}_{t-1}\|_2^2 \cdot 0.02
$$
Penalizza cambi bruschi di direzione/intensità, non la magnitudine assoluta — altrimenti scoraggerebbe anche movimenti grandi legittimi a inizio episodio quando si è ancora lontani dal target.

### 7.6 Bonus terminale (`_terminal_bonus`)

Applicato solo alla fine dell'episodio (STOP esplicito nel ramo discreto, o sempre alla truncation nel ramo continuo), composto da due parti:

**(a) Bonus assoluto**, ancorato a una soglia fissa di "buona soluzione" ($IoU=0.5$):
$$
r_{abs} = \text{clip}\big((IoU_{final} - 0.5) \cdot 18.0,\; -12,\; +12\big)
$$

**(b) Bonus di miglioramento relativo** (introdotto in "FIX v4"), che premia il progresso *rispetto al punto di partenza dell'episodio* indipendentemente dalla difficoltà assoluta:
$$
r_{improve} = \text{clip}\big((IoU_{final} - IoU_{init}) \cdot 6.0,\; -4,\; +4\big)
$$

La motivazione documentata per (b) è cruciale dal punto di vista metodologico: con difficoltà alta il box iniziale può partire senza alcuna sovrapposizione con il target, rendendo il traguardo assoluto di $IoU=0.5$ irrealistico in 200 step — il reward risulterebbe quasi sempre fortemente negativo indipendentemente dall'impegno dell'agente, un segnale poco informativo che rischia di scoraggiare l'esplorazione proprio negli stage più difficili del curriculum. Il bonus di miglioramento relativo garantisce che un progresso genuino (anche se insufficiente a superare la soglia assoluta) riceva comunque rinforzo positivo.

### 7.7 Penalità per mancato STOP (solo ramo discreto)

`NO_STOP_PENALTY = -1.0`, applicata quando l'episodio termina per *truncation* (raggiunto `max_steps`) senza che l'agente abbia mai scelto STOP — incentiva l'agente a imparare *quando* fermarsi, non solo *dove* posizionare il box.

### 7.8 Nota storica sulla calibrazione di `STOP_BONUS_SCALE`/`CLIP`

Il codice documenta un ciclo di calibrazione empirica non banale:
- valori iniziali alti (20/15) → l'agente non impara mai a fermarsi (costo di continuare troppo basso rispetto al beneficio incerto di STOP);
- valori abbassati troppo aggressivamente (12/8) → l'agente impara a fermarsi quasi subito con un box mediocre (osservato empiricamente come `action_distribution/action_8` salire a ~0.30 e `mean_ep_length` crollare a 1–3 step) perché diventa economico rispetto al costo cumulato di `TIME_PENALTY`;
- valore finale scelto (18/12), un compromesso intermedio tra i due estremi patologici.

Questo è un esempio concreto di come il reward shaping in RL applicato sia un processo iterativo di osservazione empirica (via TensorBoard) e correzione, non un esercizio puramente analitico a tavolino.

---

## 8. Il ramo discreto: 9 azioni, action masking e MaskablePPO

### 8.1 Spazio delle azioni discreto

| ID | Nome | Effetto |
|---|---|---|
| 0 | LEFT | $c_x \mathrel{-}= \max(0.05w,\, 2)$ |
| 1 | RIGHT | $c_x \mathrel{+}= \max(0.05w,\, 2)$ |
| 2 | UP | $c_y \mathrel{-}= \max(0.05h,\, 2)$ |
| 3 | DOWN | $c_y \mathrel{+}= \max(0.05h,\, 2)$ |
| 4 | WIDER | $w \mathrel{+}= \max(0.05w,\, 2)$ |
| 5 | NARROWER | $w \mathrel{-}= \max(0.05w,\, 2)$ |
| 6 | TALLER | $h \mathrel{+}= \max(0.05h,\, 2)$ |
| 7 | SHORTER | $h \mathrel{-}= \max(0.05h,\, 2)$ |
| 8 | STOP | termina l'episodio, calcola il bonus terminale |

Il passo di movimento/resize è proporzionale alla dimensione corrente del box (`step_frac`, di default 0.05, modulato dal curriculum — §11), non un valore assoluto fisso: questo rende i movimenti "relativamente" consistenti sia su tumori piccoli sia grandi.

### 8.2 Action masking

L'ambiente espone `action_masks()`, che restituisce una maschera booleana di lunghezza 9:

```python
def action_masks(self):
    mask = np.ones(N_ACTIONS, dtype=bool)
    at_w_min, at_w_max = self.w <= 12, self.w >= self.W
    at_h_min, at_h_max = self.h <= 12, self.h >= self.H
    if at_w_max: mask[4] = False   # non si può allargare oltre l'immagine
    if at_w_min: mask[5] = False   # non si può restringere sotto 12px
    if at_h_max: mask[6] = False
    if at_h_min: mask[7] = False
    mask[8] = self.current_step >= self.min_steps_before_stop  # STOP vietato troppo presto
    return mask
```

Il masking impedisce due categorie di azioni non valide o indesiderate: (a) resize oltre i limiti geometrici (ridondante col clip interno, ma evita che la policy assegni probabilità a mosse "sprecate"), e (b) STOP prematuro, controllato dal parametro `min_steps_before_stop` — anch'esso modulato dal curriculum (§11).

Tecnicamente il masking è realizzato tramite `ActionMasker` (wrapper di `sb3-contrib`) e la funzione helper `mask_fn` in `utils.py`, che semplicemente inoltra la chiamata a `env.unwrapped.action_masks()`. L'algoritmo consumatore è `MaskablePPO`, che azzera la probabilità delle azioni mascherate nel softmax della policy invece di limitarsi a penalizzarle nel reward — un approccio teoricamente più corretto per spazi di azione con vincoli strutturali noti a priori (Huang & Ontañón, 2022, "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms").

### 8.3 MaskablePPO: teoria e iperparametri

**Proximal Policy Optimization** (Schulman et al., 2017) è un algoritmo *on-policy* actor-critic che ottimizza un obiettivo surrogato clippato per limitare l'ampiezza degli aggiornamenti di policy ad ogni iterazione, evitando i collassi di performance tipici del *vanilla policy gradient*:

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\Big[\min\big(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\big)\Big]
$$

dove $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{old}}(a_t|s_t)$ è il rapporto di importanza e $\hat{A}_t$ è la stima del vantaggio, calcolata con **Generalized Advantage Estimation** (Schulman et al., 2016) con $\lambda = 0.95$.

**MaskablePPO** (`sb3-contrib`) è un'estensione che integra il masking delle azioni direttamente nella distribuzione categoriale della policy, sia in fase di rollout (campionamento) sia in fase di calcolo della loss (le azioni mascherate non contribuiscono al gradiente).

Configurazione usata (da `train.py`):

```python
model = MaskablePPO(
    policy="MultiInputPolicy",
    policy_kwargs=dict(
        features_extractor_kwargs=dict(cnn_output_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    ),
    learning_rate=linear_schedule(1.5e-4, 1e-5),   # decadimento lineare
    n_steps=1024,          # step di rollout per ambiente prima di ogni update
    batch_size=1024,
    n_epochs=6,            # epoche di ottimizzazione per rollout
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=linear_schedule(0.1, 0.03),          # decadimento lineare
    ent_coef=0.03,         # gestito/sovrascritto dinamicamente dal curriculum (§11)
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=0.03,
)
```

Con `n_envs=6` ambienti paralleli (`SubprocVecEnv`) e `n_steps=1024`, il buffer di rollout raccoglie $6 \times 1024 = 6144$ transizioni per update — una dimensione aumentata rispetto a una versione precedente ($n\_steps=512$, buffer $3072$) proprio per ridurre il rumore delle stime di vantaggio/valore, osservato empiricamente come oscillazioni eccessive di `train/explained_variance`.

`target_kl=0.03` implementa un early-stopping delle epoche di ottimizzazione se la divergenza KL tra policy vecchia e nuova supera la soglia, una salvaguardia aggiuntiva contro update troppo aggressivi anche all'interno del clipping PPO standard.

`vf_coef=0.5` (standard PPO) è stato scelto al posto di un valore più aggressivo (1.0, usato in una versione precedente) per evitare che la loss del value function domini il gradiente complessivo rispetto alla loss di policy.

### 8.4 Normalizzazione osservazioni/reward

L'ambiente vettorializzato è avvolto in `VecNormalize(norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99)`. Da notare `norm_obs=False`: le osservazioni immagine (già in `[0,255]` uint8, gestite dalla CNN) non vengono normalizzate da `VecNormalize` (sarebbe ridondante/dannoso mescolare la normalizzazione per-canale di `VecNormalize`, pensata per vettori di feature continue, con un input immagine); solo il reward viene normalizzato con una media mobile della deviazione standard, con clipping a $\pm 10$ per contenere outlier.

---

## 9. Il ramo continuo: azioni `[dx,dy,dw,dh]` e SAC

### 9.1 Motivazione ("FIX v7")

Il codice documenta esplicitamente il ragionamento che ha portato a introdurre questa seconda formulazione, individuando un **ceiling strutturale** nel ramo discreto: con azioni a passo fisso (per quanto scalato da `step_frac`), la precisione massima raggiungibile è intrinsecamente limitata dalla granularità del passo — l'agente non può convergere su un box che richiederebbe un aggiustamento più fine dello step corrente. Inoltre, l'intera classe di comportamenti patologici legati a STOP (fermarsi troppo presto, non fermarsi mai, fragilità del reward shaping associato, necessità di uno `StopCurriculumCallback` dedicato) esiste *solo* perché STOP è un'azione discreta appresa.

La soluzione adottata è duplice:
1. sostituire le 9 azioni discrete con un **vettore continuo** $[\Delta c_x, \Delta c_y, \Delta w, \Delta h] \in [-1,1]^4$ — regressione diretta del delta, non un "nudge" a passo fisso;
2. **rimuovere l'azione STOP**: l'episodio dura sempre `max_steps`, eliminando la necessità di imparare "quando fermarsi" come decisione separata.

### 9.2 Step continuo (`_step_continuous`)

```python
action = clip(action, -1, 1)
bs_x, bs_y, bs_w, bs_h = max(step_frac * w, 1), max(step_frac * h, 1), ...  # come nel ramo discreto
cx += action[0] * bs_x
cy += action[1] * bs_y
w  += action[2] * bs_w
h  += action[3] * bs_h
# clip geometrico come nel ramo discreto
```

Il reward riusa esattamente le stesse componenti del ramo discreto (`delta_iou`, `delta_dist`, `time_penalty`, `oversize_penalty`), sostituendo l'oscillazione discreta con la penalità di smoothness continua (§7.5) e applicando il bonus terminale **sempre** alla truncation, non condizionatamente a una scelta esplicita dell'agente.

### 9.3 Soft Actor-Critic: teoria e iperparametri

**SAC** (Haarnoja et al., 2018) è un algoritmo *off-policy*, actor-critic, per spazi di azione continui, basato sul framework del *maximum entropy RL*: l'obiettivo non è solo massimizzare il ritorno atteso ma anche l'entropia della policy, incoraggiando esplorazione persistente:

$$
J(\pi) = \sum_t \mathbb{E}_{(s_t,a_t)\sim\rho_\pi}\big[r(s_t,a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\big]
$$

Il coefficiente di temperatura $\alpha$ (in SB3, `ent_coef`) può essere **appreso automaticamente** (`ent_coef="auto"`) tramite un vincolo duale che mantiene l'entropia media vicino a un target (tipicamente $-\dim(\mathcal{A})$), eliminando la necessità di uno scheduling manuale — un punto esplicitamente valorizzato nei commenti del codice come motivo di preferenza rispetto a PPO per questo dominio ("elimina l'intera classe di problemi di scheduling manuale dell'entropia che ha causato metà dei bug di questa conversazione").

SAC è **off-policy**: mantiene un *replay buffer* di transizioni passate e le riusa per gli aggiornamenti (invece di scartarle dopo ogni rollout come PPO), tipicamente molto più efficiente in termini di campioni per problemi di controllo continuo fine.

Configurazione usata:

```python
policy_kwargs = dict(
    features_extractor_kwargs=dict(cnn_output_dim=512),
    net_arch=dict(pi=[256, 256], qf=[256, 256]),
)
model = SAC(
    policy="MultiInputPolicy", policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    buffer_size=15_000,            # vedi §9.4 per la motivazione del valore ridotto
    learning_starts=5_000,
    batch_size=256,
    tau=0.005,                     # soft update dei target network
    gamma=0.99,
    train_freq=1, gradient_steps=1,
    ent_coef="auto",
    optimize_memory_usage=False,   # vedi §9.4
    replay_buffer_kwargs=dict(handle_timeout_termination=False),
)
```

`tau=0.005` controlla il tasso di aggiornamento *soft* (Polyak averaging) dei target network di critic, uno standard per SAC/TD3 che stabilizza il bootstrap.

### 9.4 Vincolo di memoria del replay buffer con osservazioni Dict

Un punto tecnico rilevante, ampiamente documentato nei commenti: SB3 richiede `DictReplayBuffer` per spazi di osservazione `Dict` (necessari qui per la coppia `image`/`box_vec`), e **`DictReplayBuffer` non supporta `optimize_memory_usage=True`** (SB3 solleva un `AssertionError` esplicito se lo si lascia attivo). La disattivazione di questa ottimizzazione fa sì che `observations` e `next_observations` vengano allocate come **due array separati** invece di un unico buffer circolare riusato, **raddoppiando** la RAM effettivamente richiesta a parità di `buffer_size` nominale rispetto a quanto ci si aspetterebbe da un buffer di transizioni standard. Per questo motivo il default di `--sac-buffer-size` è stato abbassato da 50.000 a **15.000** (~5.6 GiB totali con osservazioni immagine a 4 canali 224×224 uint8, ~196KB per singola osservazione).

Analogamente, `handle_timeout_termination` — un parametro del `ReplayBuffer` sottostante, non un argomento diretto del costruttore `SAC.__init__` — va passato tramite `replay_buffer_kwargs` (un errore comune che il codice documenta come causa di un `TypeError` incontrato durante lo sviluppo). È impostato a `False` perché l'ambiente non usa `gym.wrappers.TimeLimit` e quindi non emette la chiave `info["TimeLimit.truncated"]` su cui questo flag farebbe leva.

### 9.5 Differenze di callback rispetto al ramo discreto

- **`EvalCallback`** standard al posto di `MaskableEvalCallback` (SAC non ha masking).
- **Nessun `StopCurriculumCallback`**: non esiste più un'azione STOP da schedulare.
- **`AdaptiveCurriculumCallback(manage_entropy=False)`**: SAC gestisce autonomamente l'entropia (`ent_coef="auto"`), quindi il callback di curriculum continua a gestire difficoltà e `step_frac` ma smette di scrivere sopra un valore di `ent_coef` calcolato per PPO.
- **`VisionMetricsCallback(continuous=True)`** e **GradCAM disattivata**: la GradCAM del progetto usa `model.policy.predict_values` e un `features_extractor` unificato, entrambi specifici delle `ActorCriticPolicy` di PPO — `SACPolicy` ha strutture actor/critic separate e non espone questi metodi, quindi la generazione della heatmap viene saltata esplicitamente (early-return) nel ramo continuo, sia in `callbacks.py` sia in `visual_evaluator.py` (`compute_gradcam`).

---

## 10. Il warm-start supervisionato: `localizer.py`

### 10.1 Motivazione

Localizzare un tumore partendo da un box completamente casuale in 200 step è un problema molto più difficile che *rifinire* un box già approssimativamente vicino (partendo, ad esempio, da $IoU \approx 0.5$). Il modulo introduce un piccolo **regressore CNN supervisionato**, `BoxRegressorCNN`, che predice direttamente $(c_x, c_y, w, h)$ normalizzati da una singola immagine, da usare come punto di partenza per l'agente RL invece di un box casuale.

### 10.2 Non è data leakage

Il regressore è allenato su coppie (immagine, box derivato dalla maschera) del train/val set — le stesse maschere già usate per calcolare il reward, quindi nessun dato aggiuntivo. A **inferenza** (sia durante il training RL sia nella valutazione finale sul test set) usa esclusivamente l'immagine, esattamente lo scenario di un utilizzo in produzione su un'immagine nuova priva di maschera.

### 10.3 Architettura

```python
class BoxRegressorCNN(nn.Module):
    features = Sequential(
        Conv2d(C, 32, 5, stride=2), BatchNorm2d(32), ReLU,
        Conv2d(32, 64, 3, stride=2), BatchNorm2d(64), ReLU,
        Conv2d(64, 128, 3, stride=2), BatchNorm2d(128), ReLU,
        Conv2d(128, 128, 3, stride=2), BatchNorm2d(128), ReLU,
        AdaptiveAvgPool2d(1),
    )
    head = Sequential(Flatten(), Linear(128, 64), ReLU, Linear(64, 4))
    forward: sigmoid(head(features(x)))   # cx,cy,w,h in [0,1]
```

Un backbone deliberatamente piccolo e veloce (4 blocchi conv con stride 2, global average pooling, testa MLP a 2 strati): non deve essere accurato al pixel, deve solo dare all'RL un punto di partenza sensibilmente migliore del casuale.

### 10.4 Funzione di loss

Combinazione di **Smooth L1** (Huber loss, robusta agli outlier) sulle coordinate dirette e **1 − IoU differenziabile** (calcolata convertendo $(c_x,c_y,w,h) \to (x_1,y_1,x_2,y_2)$ e applicando intersezione/unione in forma tensoriale):

$$
\mathcal{L} = \text{SmoothL1}(\hat{b}, b) + \big(1 - IoU(\hat{b}, b)\big)
$$

La componente IoU guida direttamente la metrica di interesse finale, non solo la vicinanza euclidea delle coordinate.

### 10.5 Training e selezione del modello

`train_localizer(train_ds, val_ds, save_path, epochs=15, batch_size=32, lr=1e-3)`: ottimizzatore Adam, il checkpoint viene salvato solo quando la IoU di validazione migliora (early best-model selection), stampando per epoca `train_loss` e `val_iou`.

### 10.6 Inferenza forzata su CPU dentro i sotto-processi

Un dettaglio ingegneristico rilevante: quando il localizzatore è usato come warm-start dentro `SubprocVecEnv` (un processo per ambiente parallelo), il caricamento è **forzato su CPU** (`load_localizer(path, device="cpu")`), anche se una GPU è disponibile. La motivazione documentata: se ogni sotto-processo dovesse inizializzare un proprio contesto CUDA per un modello minuscolo, si genererebbe inutile contesa di memoria/rallentamenti — l'inferenza qui è un singolo forward pass su un'immagine 224×224, per cui la CPU è più che sufficiente.

---

## 11. Curriculum learning: dalla rampa fissa all'auto-paced

Il progetto implementa **quattro generazioni progressive** di callback per il curriculum, tutte presenti nel codice (le prime tre restano disponibili come classi ma non sono più usate in `train.py`, sostituite dall'ultima). Questa evoluzione è di per sé un caso di studio istruttivo sul perché il curriculum learning "naive" a tempo fisso è fragile in pratica.

### 11.1 Generazione 1: rampe lineari indipendenti

- **`EntropyScheduleCallback`**: decadimento lineare di `ent_coef` da un valore iniziale a un valore finale su `schedule_steps`.
- **`StopCurriculumCallback`**: decadimento lineare di `min_steps_before_stop` da `MAX_STEPS_PER_EPISODE+1` (STOP impossibile all'inizio) a 0 (STOP sempre permesso), su `curriculum_steps` fissi. **Ancora attivo** in `train.py` per il ramo discreto (`STOP_CURRICULUM_STEPS = 350_000`), perché "quando posso iniziare a valutare se fermarmi" è considerata una competenza generale, non legata allo stage di difficoltà del box.
- **`StepSizeScheduleCallback`**: decadimento lineare di `step_frac` (passi via via più fini man mano che si procede).
- **`InitBoxCurriculumCallback`**: rampa lineare *continua* di `init_difficulty` da 0 a 1 su `curriculum_steps` step.

### 11.2 Generazione 2: `StagedCurriculumCallback` (a gradini con reheat)

Sostituisce la rampa continua con **stage discreti** (difficoltà costante per blocchi lunghi), motivato dal fatto che con una rampa continua l'ambiente non è mai stazionario — la policy non fa in tempo a consolidarsi prima che la difficoltà cambi di nuovo sotto di lei.

Introduce inoltre il meccanismo di **"reheat" dell'entropia**: ad ogni cambio di stage, `ent_coef` torna temporaneamente a un valore alto (`reheat_ent`) e decade linearmente verso un pavimento (`floor_ent`) entro una frazione dello stage (`reheat_frac=0.35`), prima del prossimo salto di difficoltà. Il razionale: un pavimento di entropia fisso per tutto il training è o troppo alto quando servirebbe sfruttare ciò che si è imparato, o troppo basso quando arrivano scenari più difficili che richiederebbero esplorazione fresca — un profilo "a dente di sega" sincronizzato con le transizioni di stage risolve entrambi i problemi.

### 11.3 Generazione 3 → 4: da a-tempo-fisso ad **auto-paced**

Il limite della Generazione 2 (e delle rampe a tempo fisso in generale) è che la difficoltà avanza indipendentemente da quanto l'agente abbia effettivamente imparato lo stage corrente. Il progetto risolve questo con **`AdaptiveCurriculumCallback`**, l'unico curriculum callback effettivamente istanziato in `train.py` nella versione corrente, che rende la progressione **gated dalle performance**:

```python
AdaptiveCurriculumCallback(
    n_stages=6, initial_difficulty=0.0, final_difficulty=1.0,
    window=300, min_steps_per_stage=250_000,
    advance_threshold=0.55, regress_threshold=0.20,
    stall_patience=300_000,
    reheat_ent=0.03, floor_ent=0.01,
    reheat_step_frac=0.05, floor_step_frac=0.012,
    reheat_duration=60_000,
    manage_entropy=not args.continuous,   # False per SAC (§9.5)
)
```

Logica di transizione (`_on_step`):
1. Mantiene un buffer a finestra scorrevole (`deque(maxlen=window)`) delle IoU finali degli ultimi episodi.
2. Se il buffer è pieno **e** sono trascorsi almeno `min_steps_per_stage` dall'ultima transizione:
   - **avanza** di uno stage se il success rate (frazione di episodi con $IoU_{final} \geq 0.5$) $\geq$ `advance_threshold` (0.55);
   - **regredisce** di uno stage (mai sotto lo stage 0) se il success rate $<$ `regress_threshold` (0.20) — se la difficoltà è davvero eccessiva, il curriculum torna indietro invece di lasciare l'agente "marcire" in una condizione irrecuperabile.
3. Ogni transizione (avanti o indietro) azzera il buffer e fa scattare un **reheat** di `ent_coef` e `step_frac` (tornano ai valori alti e decadono linearmente in `reheat_duration=60_000` step).
4. Se lo stage resta invariato per più di `stall_patience=300_000` step, scatta comunque un reheat periodico ("nuovo tentativo" con più esplorazione) — **anche se si è già all'ultimo stage**, correggendo esplicitamente un bug della Generazione 2 in cui il reheat non si ripeteva più una volta raggiunto lo stage finale, lasciando la fase più difficile priva di esplorazione fresca per il resto del training.

### 11.4 Due patologie osservate e corrette empiricamente (documentate nei commenti)

- **"FIX v5" — flapping tra stage**: con il warm-start del localizzatore attivo, gli episodi terminano molto più rapidamente; un buffer piccolo (`window=100` nella versione precedente) si riempiva in pochissimi step, ben prima del vecchio `min_steps_per_stage=30_000`. Poiché il buffer viene svuotato ad ogni transizione, e subito dopo una regressione il success rate schizza meccanicamente in alto (il compito è improvvisamente più facile, non perché si sia imparato di più), il sistema poteva ri-avanzare quasi subito — un ping-pong perenne tra due stage senza mai consolidare, che oltretutto riaccendeva il reheat ad ogni transizione impedendo all'entropia di scendere mai al pavimento. Soluzione: `window` e `min_steps_per_stage` alzati sensibilmente (100→300 episodi, 30.000→150.000 step) per dare tempo alle statistiche di assestarsi.
- **"FIX v6" — avanzamento troppo permissivo**: risolto il flapping, è emerso che il curriculum avanzava esattamente ogni `min_steps_per_stage`, a **ogni** transizione, perché la soglia precedente (`advance_threshold=0.35`) veniva quasi sempre soddisfatta appena scaduto il tempo minimo — di fatto un curriculum "adattivo" degenerato in un curriculum a tempo fisso con periodo 150k, con la difficoltà che saliva più in fretta della vera qualità appresa (osservabile come calo strutturale della IoU assoluta ad ogni nuovo stage nella metrica TensorBoard `custom_plots/1_iou_final`). Soluzione: soglie di avanzamento/regressione alzate (0.35→0.55, 0.08→0.20) per richiedere vera padronanza prima di avanzare, con `min_steps_per_stage` alzato di conseguenza (150k→250k).

Questa cronologia è un esempio concreto e ben documentato di **debug empirico guidato da telemetria** in un progetto di RL applicato — le decisioni di parametrizzazione non sono arbitrarie ma la conseguenza diretta dell'osservazione di comportamenti degeneri specifici su TensorBoard.

---

## 12. Callback di monitoraggio e spiegabilità (GradCAM)

### 12.1 `VisionMetricsCallback`

Il callback centrale di logging, con responsabilità multiple:

- **Layout custom di TensorBoard**: alla `_on_training_start`, registra un layout con scalari multilinea raggruppati (`add_custom_scalars`) in sezioni tematiche: IoU media/std/finale, guida alla risoluzione (delta IoU/distanza), reward completo, curriculum (ent_coef, min_steps_before_stop, step_frac).
- **Log delle componenti di reward** ad ogni step (`_on_step`): media su tutti gli ambienti paralleli di `delta_iou`, `delta_dist`, `total`, `oversize_penalty`, `oscillation_penalty`.
- **Log delle metriche di episodio**: IoU media/std/finale, success rate (`ep_iou_final >= 0.5`).
- **Distribuzione delle azioni** (solo ramo discreto): istogramma di frequenza delle 9 azioni, campionato ogni 50.000 step, utile per diagnosticare collassi di policy (es. l'agente che sceglie sempre STOP, osservato durante la calibrazione del reward — §7.8).
- **GradCAM periodica** (`generate_and_save_gradcam`, ogni `gradcam_every=20` rollout, sempre al primo): genera una heatmap di attivazione **Grad-CAM** (Selvaraju et al., 2017) sull'ultimo layer convoluzionale dell'estrattore CNN, mostrando quali regioni dell'immagine la value function considera più rilevanti per la stima del valore, sovrapposta a GT (verde) e box predetto dall'azione greedy corrente (rosso). Salvata come PNG in `ppo_brain_tumor_logs/.../gradcam_outputs/`. **Disattivata esplicitamente nel ramo continuo** (early-return se `self.continuous`), perché si basa su `model.policy.predict_values` e un `features_extractor` unificato, entrambi specifici delle policy PPO.

Tecnicamente la GradCAM è implementata registrando forward/backward hook su `model.policy.features_extractor.extractors["image"].cnn[4]` (il layer convoluzionale target dentro il `CombinedExtractor` del `MultiInputPolicy`), calcolando il gradiente della value function predetta rispetto alle attivazioni, pesando i canali con la media spaziale del gradiente (Global Average Pooling dei gradienti, come da formulazione originale di Grad-CAM) e normalizzando la heatmap risultante in $[0,1]$.

### 12.2 `ModelCheckpointCallback`

Si veda §13.

---

## 13. Checkpointing e gestione dello stato di training

`ModelCheckpointCallback` salva **modello e statistiche `VecNormalize` insieme**, ogni `save_freq` step (`--checkpoint-every`, default 10.000), con il numero di timestep nel nome del file (`ppo_brain_tumor_<step>.zip` + `ppo_brain_tumor_<step>_vecnormalize.pkl`). Questo è motivato esplicitamente nel codice come correzione rispetto a un comportamento originario in cui `VecNormalize` veniva salvato solo a fine training: se il processo si fosse interrotto prima (crash, timeout, interruzione manuale), le statistiche di normalizzazione sarebbero andate perse, rendendo il checkpoint del modello effettivamente inutilizzabile per un resume corretto (le statistiche di running mean/varianza del reward sono parte integrante dello stato della policy addestrata).

Viene mantenuta una coda degli ultimi `keep_last=10` checkpoint, con cancellazione automatica dei più vecchi per contenere l'occupazione disco (osservabile nella cartella `ppo_brain_tumor_logs/1/checkpoints/`, che nella repository analizzata contiene infatti esattamente le coppie modello+vecnormalize per un insieme di step consecutivi, coerente con questa politica di retention).

Il ciclo principale in `train.py` chiama `model.learn(..., reset_num_timesteps=(i==1))` per `n_iterations` volte consecutive, salvando anche uno snapshot cumulativo (`{total_timesteps*i}.zip`) ad ogni iterazione — il `reset_num_timesteps=False` per $i>1$ è essenziale per evitare che Stable-Baselines3 sovrascriva la stessa cartella TensorBoard (`MaskablePPO_0`) ad ogni riavvio dello script invece di crearne una nuova incrementale (si notino infatti, nella struttura `tb/` del progetto, le cartelle `MaskablePPO_0` … `MaskablePPO_8` e `SAC_1`, prova diretta di questo comportamento nella repository).

---

## 14. Valutazione quantitativa e visiva: `visual_evaluator.py`

### 14.1 `VisualEvaluator`: rollout ispezionabili

- **`watch(idx)`**: apre una finestra `cv2.imshow` live sull'episodio relativo al campione `idx` del test set (richiede display).
- **`save_episode_video(idx)`**: registra un video (MP4 via `cv2.VideoWriter`, o GIF via `imageio`) con un frame per step, box verde/rosso e HUD — utile in ambienti headless.
- **`save_episode_steps_grid(idx)`**: un'unica immagine con `n_frames=6` frame chiave campionati uniformemente lungo l'episodio (inizio, step intermedi, fine), affiancati orizzontalmente.

Il rollout condiviso (`_rollout_with_frames`) gestisce trasparentemente sia il ramo discreto (con `action_masks()` passato a `model.predict`) sia il ramo continuo (nessun masking) tramite il flag `continuous_actions` passato al costruttore — un punto che il codice documenta essere stato **sorgente di un bug concreto**: prima della correzione, `continuous_actions` non veniva propagato da `train.py` all'`Evaluator`, causando un crash certo ("il modello SAC non accetta il parametro `action_masks`") sia in modalità `--eval-only`/`--watch-idx` sia — più insidiosamente — **a fine training completo**, subito dopo aver atteso tutte le `n_iterations` di `model.learn()`.

### 14.2 `evaluate_dataset`: valutazione aggregata sul test set

Per ogni campione del test set (di default tutti, `max_samples=0`):
1. Esegue il rollout completo, calcola metriche geometriche (`_box_metrics`): IoU, area di intersezione, `coverage_ratio` (intersezione / area GT), `size_ratio` (area predetta / area GT).
2. Classifica il campione per **bucket dimensionale** (`_size_bucket`, soglie `SIZE_BUCKET_EDGES=(0.10, 0.20)`: small/medium/large in base a `gt_area_ratio`) e **bucket di contrasto** (`_intensity_bucket`: bright/dark, confrontando l'intensità media dentro il box GT con la media globale).
3. Salva l'ultimo frame come immagine box (`test_eval/boxes/`), e — solo nel ramo discreto — una heatmap GradCAM analoga a quella di training (`test_eval/gradcam/`, via `compute_gradcam` in `visual_evaluator.py`).
4. Scrive una riga CSV (`metrics_per_sample.csv`) con tutte le metriche per campione.

Al termine:
- **Salvataggio dei casi peggiori**: i 20 campioni con IoU più bassa (`n_failure_cases=20`) vengono copiati in `failure_cases/` con nome che include il rank e la IoU; per i 5 peggiori in assoluto (`save_videos_for_worst=5`) viene anche generato il video completo dell'episodio, per ispezionare esattamente cosa sia andato storto passo per passo.
- **Grafici aggregati** (`_make_plots`, Matplotlib, backend `Agg` per compatibilità headless):
  1. istogramma della distribuzione di IoU finale, con linea verticale alla soglia di successo;
  2. barre di IoU media per bucket dimensionale e per bucket di contrasto, con conteggio campioni annotato su ogni barra;
  3. curva media di IoU e reward **nel tempo** (allineando tutti gli episodi per indice di step, con `np.nanmean` per gestire episodi di lunghezza diversa nel ramo discreto), su doppio asse Y;
  4. grafico a torta del success rate globale ($IoU \geq 0.5$).
- **`summary.txt`**: riepilogo testuale con IoU media/mediana/std, percentuale di campioni sopra ciascuna soglia (`IOU_THRESHOLDS = (0.3, 0.5, 0.7)`), success rate, percentuale di episodi terminati con STOP esplicito vs timeout (solo significativo nel ramo discreto), e breakdown per bucket dimensionale/contrasto.

Questa combinazione di metriche aggregate + breakdown per sotto-popolazione + ispezione diretta dei casi peggiori costituisce una metodologia di valutazione robusta, in linea con le buone pratiche di *error analysis* raccomandate in letteratura per sistemi di visione applicati a domini medici (dove la performance media può mascherare fallimenti sistematici su sottogruppi specifici, ad esempio tumori molto piccoli o a basso contrasto).

---

## 15. Il flusso end-to-end di `train.py`

1. **Parsing CLI** (`build_arg_parser`) e costruzione della configurazione dataset (`build_dataset_config`), con fallback a variabili d'ambiente (`DATASET_SOURCE`, `DATASET_PATH`, `KAGGLE_DATASET_ID`, `TOTAL_TIMESTEPS`) se gli argomenti CLI non sono specificati.
2. **Caricamento dataset** (`get_datasets(cfg)`), con dispatch automatico kaggle/local/synthetic.
3. **(Opzionale) training del localizzatore supervisionato** (`--train-localizer`), poi **(opzionale) caricamento come warm-start** (`--use-localizer`), forzato su CPU per compatibilità coi sotto-processi.
4. **Modalità alternative** (`--eval-only` / `--watch-idx`): carica un modello e statistiche `VecNormalize` da checkpoint e salta direttamente alla valutazione visiva, senza rifare training.
5. **Sanity check dell'ambiente** (`check_env`, utility di SB3 che verifica la conformità Gymnasium prima di investire tempo in training).
6. **Costruzione degli ambienti vettorializzati**:
   - training: `SubprocVecEnv` (parallelismo multi-processo reale, non solo multi-thread) con `N_ENVS=6` istanze, avvolto in `VecMonitor` (logging automatico di lunghezza/reward per episodio) e `VecNormalize` (solo reward);
   - valutazione: `DummyVecEnv` singolo, `VecNormalize(training=False, norm_reward=False)` — le statistiche di normalizzazione **non vengono aggiornate** durante la valutazione, solo applicate (comportamento corretto per valutazione out-of-distribution rispetto al training corrente).
7. **Istanziazione dei callback**: `VisionMetricsCallback`, `AdaptiveCurriculumCallback`, (solo discreto) `StopCurriculumCallback`, `ModelCheckpointCallback`, e infine `EvalCallback`/`MaskableEvalCallback` con `StopTrainingOnNoModelImprovement(max_no_improvement_evals=120, min_evals=180)` come callback-dopo-valutazione — una pazienza volutamente alta, motivata dal fatto che l'`eval_env` valuta di default a `init_difficulty=1.0` (il caso più difficile) mentre il train-env procede secondo il curriculum da 0: per gran parte del training la valutazione appare "piatta" semplicemente perché il compito valutato è più difficile di quanto l'agente abbia ancora esercitato, non perché abbia smesso di migliorare — con pazienza troppo bassa questo avrebbe fatto scattare uno stop anticipato ingiustificato (episodio concreto documentato: intorno a 1.1M step in una run analizzata).
8. **Istanziazione dell'algoritmo** (SAC o MaskablePPO, §8.3/§9.3).
9. **Ciclo di training**: `for i in range(1, n_iterations+1): model.learn(total_timesteps, callback=callbacks, reset_num_timesteps=(i==1)); model.save(...)`.
10. **Salvataggio finale** delle statistiche `VecNormalize` complessive.
11. **Valutazione visiva finale** sul test set completo (`VisualEvaluator.evaluate_dataset(max_samples=0)`).

---

## 16. Interfaccia a riga di comando e casi d'uso

```bash
# training completo, sorgente Kaggle, 110 iterazioni da 100k step ciascuna (11M step totali)
python train.py --total-timesteps 100000 --dataset-source kaggle --n-iterations 110

# training con warm-start supervisionato
python train.py --train-localizer --use-localizer --localizer-epochs 15

# training in modalità continua (SAC)
python train.py --continuous --sac-buffer-size 15000 --sac-learning-starts 5000

# solo valutazione visiva di un checkpoint già allenato
python train.py --eval-only \
  --model-path ./ppo_brain_tumor_logs/checkpoints/ppo_brain_tumor_500000.zip \
  --vecnorm-path ./ppo_brain_tumor_logs/checkpoints/ppo_brain_tumor_500000_vecnormalize.pkl

# osservare l'agente lavorare dal vivo su un campione del test set (richiede display)
python train.py --watch-idx 3 --model-path ... --vecnorm-path ...
```

Parametri principali esposti: `--n-envs` (default 6), `--n-epochs` (default 6, solo PPO), `--checkpoint-every` (default 10.000), `--output-root` (default `./ppo_brain_tumor_logs`).

---

## 17. Cronologia evolutiva del progetto (FIX v2 → v7)

Ricostruita dai commenti storici lasciati deliberatamente nel codice (una pratica di documentazione particolarmente preziosa per un progetto sperimentale iterativo):

| Versione | Problema osservato | Correzione |
|---|---|---|
| v2 | Rollout buffer piccolo (`n_steps=512`) → stime di vantaggio/valore rumorose (`explained_variance` instabile) | `n_steps` alzato a 1024 |
| v3 | — | `ent_coef` allineato al valore di reheat del curriculum a stage; `vf_coef` abbassato da 1.0 a 0.5 |
| v4 | Curriculum a step_count fisso avanza indipendentemente dalle performance reali; a difficoltà alta il traguardo assoluto IoU=0.5 è irrealistico, reward poco informativo | Introdotto bonus di miglioramento relativo (`IMPROVEMENT_BONUS_*`); introdotto `AdaptiveCurriculumCallback` auto-paced |
| v5 | Flapping tra stage adiacenti del curriculum (buffer piccolo si svuota troppo in fretta col warm-start, decisioni premature su rumore statistico) | `window` 100→300, `min_steps_per_stage` 30k→150k |
| v6 | Curriculum avanzava ad *ogni* transizione esattamente al tempo minimo consentito (soglie troppo permissive, degenerato in curriculum a tempo fisso) | `advance_threshold` 0.35→0.55, `regress_threshold` 0.08→0.20, `min_steps_per_stage` 150k→250k |
| v7 | Ceiling strutturale del ramo discreto (precisione limitata dal passo fisso); intera classe di bug legati al timing di STOP | Introdotto ramo continuo `[dx,dy,dw,dh]` + SAC, rimozione di STOP, `Dict` observation space (`image`+`box_vec`) al posto dei piani di coordinate, gestione della RAM del `DictReplayBuffer` |

A questa lista si affiancano correzioni puntuali documentate nel codice ma non necessariamente numerate: allineamento di `init_difficulty` tra train-env ed eval-env (altrimenti la valutazione appariva artificiosamente stagnante nelle prime centinaia di migliaia di step); propagazione mancante di `continuous_actions` all'`Evaluator` (causa di crash sia in `--eval-only` sia a fine training); calibrazione empirica di `STOP_BONUS_SCALE`/`CLIP` (§7.8); correzione della `FallbackTransform` che ignorava la configurazione di augmentation e applicava sempre un flip verticale anatomicamente implausibile.

---

## 18. Analisi critica, limiti e lavori futuri

### 18.1 Punti di forza

- **Documentazione storica nel codice** eccezionalmente ricca, che rende il progetto un caso di studio prezioso sul processo iterativo di debug in RL applicato, non solo sul risultato finale.
- **Doppia formulazione discreta/continua** dello stesso problema, con motivazioni tecniche solide per entrambe e gestione pulita delle differenze algoritmiche (masking, GradCAM, gestione entropia) tramite flag condivisi.
- **Curriculum adattivo gated dalle performance**, un miglioramento concettualmente solido rispetto a rampe a tempo fisso, con meccanismi espliciti sia di avanzamento sia di regressione.
- **Warm-start supervisionato senza data leakage**, con attenzione esplicita a mantenere la separazione train/test anche nella scelta della scala del rumore di inizializzazione.
- **Valutazione stratificata per dimensione/contrasto del tumore**, non solo metrica aggregata, con ispezione diretta dei casi di fallimento.

### 18.2 Limiti e possibili estensioni

- Il reward shaping, per quanto ben calibrato empiricamente, resta una combinazione lineare di termini euristici; un confronto sistematico con formulazioni alternative (ad esempio reward shaping potential-based in senso stretto, Ng et al. 1999, con garanzie teoriche di invarianza della policy ottima) non risulta presente nel codice analizzato.
- Il regressore di warm-start e l'agente RL sono allenati separatamente e in sequenza; un'estensione naturale sarebbe l'allenamento congiunto o l'uso del localizzatore anche come *baseline* di value function iniziale (non solo come punto di partenza geometrico).
- Il confronto quantitativo diretto tra il ramo discreto (MaskablePPO) e continuo (SAC) — in termini di sample efficiency, IoU finale media e stabilità — non è automatizzato nel codice attuale: richiederebbe un harness di valutazione comparativa dedicato (stesso test set, stesso budget di step, metriche affiancate).
- L'assenza, nell'ambiente, di un vincolo esplicito di forma del box (es. rapporto d'aspetto minimo/massimo) lascia in teoria spazio a soluzioni degeneri (box estremamente sottili) che il solo termine di oversize penalty non copre esplicitamente.
- La GradCAM, disattivata nel ramo continuo per incompatibilità strutturale con SAC, potrebbe essere sostituita da una tecnica di spiegabilità applicabile anche a policy actor-critic con reti actor/critic separate (ad esempio Integrated Gradients sull'attore).

---

## 19. Conclusioni

Il progetto `segmentation_rl` costituisce un'implementazione matura e ben strumentata di un sistema di localizzazione sequenziale di tumori cerebrali via reinforcement learning, che affronta un problema di visione medica con un approccio alternativo alla segmentazione diretta, enfatizzando interpretabilità del processo decisionale e adattabilità progressiva della difficoltà del compito. La compresenza di due formulazioni (discreta con MaskablePPO e continua con SAC), entrambe accompagnate da motivazioni tecniche esplicite e da una storia di iterazione documentata nel codice stesso, rende il progetto un riferimento utile non solo per il risultato applicativo ma anche come esempio metodologico di ingegneria del reward, del curriculum e della gestione delle risorse computazionali (memoria dei replay buffer, parallelismo multi-processo, warm-start supervisionato) in un sistema di RL applicato a dati reali.

---

## 20. Bibliografia

1. Stember, J.N., Shalu, H.. Reinforcement learning using Deep networks and learning accurately localizes brain tumors on MRI with very small training sets (2022). DOI 10.1186/s12880-022-00919-x, https://doi.org/10.1186/s12880-022-00919-x
2. Ding, Yi and Qin, Xue and Zhang, Mingfeng and Geng, Ji and Chen, Dajiang and Deng, Fuhu and Song, Chunhe. RLSegNet: An Medical Image Segmentation Network Based on Reinforcement Learning (2023). DOI 10.1109/TCBB.2022.3195705, https://ieeexplore-ieee-org.unimib.idm.oclc.org/abstract/document/9847069
3. Joseph Stember, Hrithwik Shalu. Deep reinforcement learning to detect brain lesions on MRI: a proof-of-concept application of reinforcement learning to medical images (2008). DOI 10.48550/arXiv.2008.02708, https://doi.org/10.48550/arXiv.2008.02708
4. Juan C. Caicedo, Svetlana Lazebnik. Active Object Localization with Deep Reinforcement Learning (2015). DOI 10.48550/arXiv.1511.06015, https://doi.org/10.48550/arXiv.1511.06015

Da togliere se non necessari:

5. Zequn Jie, Xiaodan Liang, Jiashi Feng, Xiaojie Jin, Wen Feng Lu, Shuicheng Yan. Tree-Structured Reinforcement Learning for Sequential Object Localization (2017). DOI 10.48550/arXiv.1703.02710, https://doi.org/10.48550/arXiv.1703.02710

2. Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. ICLR.
3. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. ICML.
4. Haarnoja, T., Zhou, A., Hartikainen, K., et al. (2018). *Soft Actor-Critic Algorithms and Applications*. arXiv:1812.05905 (versione con temperatura auto-regolata).
5. Huang, S., & Ontañón, S. (2022). *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*. FLAIRS / arXiv:2006.14171.
6. Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529–533.
7. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV.
8. Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping*. ICML.
9. Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). *Curriculum Learning*. ICML.
10. Narvekar, S., Peng, B., Leonetti, M., Sinapov, J., Taylor, M. E., & Stone, P. (2020). *Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey*. JMLR.
11. Caicedo, J. C., & Lazebnik, S. (2015). *Active Object Localization with Deep Reinforcement Learning*. ICCV.
12. Bueno, M. B., Ramanishka, V., et al. (2016). *Hierarchical Object Detection with Deep Reinforcement Learning*. NeurIPS Workshop.
13. Zuiderveld, K. (1994). *Contrast Limited Adaptive Histogram Equalization*. Graphics Gems IV, Academic Press.
14. Tomasi, C., & Manduchi, R. (1998). *Bilateral Filtering for Gray and Color Images*. ICCV.
15. Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Druzhinin, M., & Kalinin, A. A. (2020). *Albumentations: Fast and Flexible Image Augmentations*. Information, 11(2), 125.
17. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI. (riferimento concettuale per il confronto con la segmentazione diretta discusso in §1)
18. Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR. (ottimizzatore usato in `localizer.py`)
19. Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR, 22(268), 1–8.
20. Towers, M., Terry, J. K., Kwiatkowski, A., et al. (2023). *Gymnasium* (successore manutenuto di OpenAI Gym), Farama Foundation.