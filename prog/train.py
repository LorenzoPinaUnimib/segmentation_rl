"""
train.py
────────
Pipeline principale: costruisce dataset/ambienti, allena MaskablePPO con
tutti i callback (curriculum + GradCAM + checkpoint completi), e alla fine
lancia la valutazione visiva sul test set.

Esempi d'uso:
    # training completo
    python train.py --total-timesteps 100000 --dataset-source kaggle

    # solo valutazione visiva di un modello già allenato
    python train.py --eval-only --model-path ./ppo_brain_tumor_logs/checkpoints/ppo_brain_tumor_500000.zip \
                     --vecnorm-path ./ppo_brain_tumor_logs/checkpoints/ppo_brain_tumor_500000_vecnormalize.pkl

    # guardare l'agente lavorare dal vivo su un campione del test set (richiede display)
    python train.py --watch-idx 3 --model-path ... --vecnorm-path ...

    # NUOVO: Simulazione manuale interattiva dell'ambiente (Discreto / PPO)
    python train.py --simulate --dataset-source synthetic

    # NUOVO: Simulazione manuale interattiva dell'ambiente (Continuo / SAC)
    python train.py --simulate --continuous --dataset-source synthetic
"""
import argparse
import os
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker

from callbacks import (
    AdaptiveCurriculumCallback,
    ModelCheckpointCallback,
    StopCurriculumCallback,
    VisionMetricsCallback,
)
from config import MAX_STEPS_PER_EPISODE
from dataset import get_datasets
from environment import BrainTumorRL_Env
from localizer import load_localizer, make_localizer_fn, train_localizer
from utils import linear_schedule, mask_fn
from visual_evaluator import VisualEvaluator


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--dataset-source", type=str, default=None, choices=["kaggle", "local", "synthetic"])
    p.add_argument("--dataset-path", type=str, default=None)
    p.add_argument("--kaggle-id", type=str, default=None)
    p.add_argument("--n-envs", type=int, default=6)
    p.add_argument("--n-epochs", type=int, default=6)
    p.add_argument("--n-iterations", type=int, default=110, help="quante volte ripetere model.learn(total_timesteps)")
    p.add_argument("--checkpoint-every", type=int, default=10_000, help="ogni quanti step salvare modello+vecnormalize")
    p.add_argument("--output-root", type=str, default="./ppo_brain_tumor_logs")

    # modalità alternative: valutazione o osservazione dal vivo senza (ri)allenare
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--watch-idx", type=int, default=None, help="indice del test set da osservare dal vivo (richiede display)")
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--vecnorm-path", type=str, default=None)

    # NUOVO: Simulazione interattiva manuale gestita dall'utente
    p.add_argument("--simulate", action="store_true", help="avvia un pannello interattivo per controllare manualmente il box e vedere i reward")

    # warm-start supervisionato (vedi localizer.py)
    p.add_argument("--train-localizer", action="store_true",
                    help="allena il regressore CNN supervisionato prima del training RL")
    p.add_argument("--use-localizer", action="store_true",
                    help="usa il localizzatore (da --localizer-path, o quello appena allenato) come warm-start per l'RL")
    p.add_argument("--localizer-path", type=str, default=None,
                    help="percorso del checkpoint del localizzatore (default: <output-root>/localizer.pt)")
    p.add_argument("--localizer-epochs", type=int, default=15)

    # FIX v7: action space continuo + SAC al posto delle 9 azioni discrete +
    # MaskablePPO. Vedi commento FIX v7 in environment.py per il perche'.
    p.add_argument("--continuous", action="store_true",
                    help="usa action space continuo [dx,dy,dw,dh] + algoritmo SAC invece delle 9 azioni discrete + MaskablePPO")
    p.add_argument("--sac-buffer-size", type=int, default=15_000,
                    help="dimensione replay buffer SAC (osservazioni immagine: tienilo basso, e' in RAM). "
                         "FIX: DictReplayBuffer (richiesto dalla Dict observation space image+box_vec) non "
                         "supporta optimize_memory_usage, quindi la RAM raddoppia rispetto a un buffer normale "
                         "(osservazioni + next_osservazioni duplicate) -- il default e' stato abbassato di "
                         "conseguenza da 50_000 a 15_000 (~5.6 GiB con osservazioni a 4 canali).")
    p.add_argument("--sac-learning-starts", type=int, default=5_000)
    return p


def build_dataset_config(args):
    dataset_source = args.dataset_source or os.environ.get("DATASET_SOURCE", "kaggle")
    dataset_local_path = args.dataset_path or os.environ.get("DATASET_PATH", None)
    kaggle_id = args.kaggle_id or os.environ.get(
        "KAGGLE_DATASET_ID", "pkdarabi/brain-tumor-image-dataset-semantic-segmentation"
    )
    if dataset_source == "local" and not dataset_local_path:
        raise SystemExit("--dataset-source=local richiede anche --dataset-path")

    return {
        "dataset": {
            "source": dataset_source, "kaggle_id": kaggle_id, "local_path": dataset_local_path,
            "image_size": [224, 224], "in_channels": 1,
            "train_ratio": (1501/2145), "val_ratio": (429/2145), "cache_pairs": False,
        },
        "preprocessing": {
            "normalization": "per_image", "binarize_mask": True, "mask_threshold": 0.5,
            "white_balance": True, "clahe": True, "denoise": False,
        },
        "training": {"batch_size": 512, "num_workers": 0},
        "output": {"root": args.output_root},
        "seed": 42,
    }


def make_train_env_fn(train_ds, initial_min_steps_before_stop, localizer_fn=None, continuous_actions=False):
    def _init():
        env = BrainTumorRL_Env(
            pytorch_dataset=train_ds, max_steps=MAX_STEPS_PER_EPISODE,
            min_steps_before_stop=initial_min_steps_before_stop, init_difficulty=0.0,
            localizer_fn=localizer_fn, continuous_actions=continuous_actions,
        )
        return env if continuous_actions else ActionMasker(env, mask_fn)
    return _init


def make_eval_env_fn(val_ds, init_difficulty=0.5, localizer_fn=None, continuous_actions=False):
    def _init():
        env = BrainTumorRL_Env(pytorch_dataset=val_ds, max_steps=MAX_STEPS_PER_EPISODE,
                                min_steps_before_stop=0, init_difficulty=init_difficulty,
                                localizer_fn=localizer_fn, continuous_actions=continuous_actions)
        return env if continuous_actions else ActionMasker(env, mask_fn)
    return _init


def load_model_and_vecenv(args, dummy_env_fn):
    if not args.model_path:
        raise SystemExit("--model-path e' richiesto per --eval-only / --watch-idx")

    raw_env = DummyVecEnv([dummy_env_fn])
    raw_env = VecMonitor(raw_env)
    if args.vecnorm_path and os.path.exists(args.vecnorm_path):
        vec_env = VecNormalize.load(args.vecnorm_path, raw_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print("[warn] nessun vecnorm-path valido: procedo senza normalizzazione (potrebbe degradare le prestazioni).")
        vec_env = raw_env

    model_cls = SAC if args.continuous else MaskablePPO
    model = model_cls.load(args.model_path)
    return model, vec_env


def run_manual_simulation(dataset, args):
    """Apre una finestra interattiva per testare manualmente l'ambiente (PPO e SAC)."""
    import cv2
    
    print("\n" + "="*75)
    print(f"[SIMULAZIONE] AVVIO SESSIONE MANUALE INTERATTIVA")
    print(f"[SIMULAZIONE] Spazio delle Azioni: {'CONTINUO (SAC)' if args.continuous else 'DISCRETO (PPO)'}")
    print("="*75)
    if not args.continuous:
        print(" CONTROLLI DA TASTIERA (Modalità Discreta PPO):")
        print("  • W / S   : Sposta Box su / giù")
        print("  • A / D   : Sposta Box sinistra / destra")
        print("  • E / Q   : Espandi / Riduci Larghezza (w)")
        print("  • R / F   : Espandi / Riduci Altezza (h)")
        print("  • SPAZIO  : Esegui azione di STOP (Termina ed emette i terminal bonus)")
    else:
        print(" CONTROLLI TRAMITE SLIDER + TASTIERA (Modalità Continua SAC):")
        print("  • Usa i 4 Slider/Trackbar OpenCV a schermo per regolare [dx, dy, dw, dh] in [-1.0, 1.0]")
        print("  • SPAZIO  : Conferma e applica il vettore impostato (Esegue lo Step)")
    print("\n CONTROLLI GENERALI:")
    print("  • N       : Forza il Reset dell'ambiente (Cambia immagine / Campione)")
    print("  • ESC / Q : Chiudi il simulatore ed esci")
    print("="*75 + "\n")

    # Inizializziamo l'ambiente a difficoltà intermedia (0.5) per il test manuale
    env = BrainTumorRL_Env(
        pytorch_dataset=dataset, max_steps=MAX_STEPS_PER_EPISODE,
        min_steps_before_stop=0, init_difficulty=0.5,
        continuous_actions=args.continuous, render_mode="rgb_array"
    )

    window_name = "Simulatore Manuale Ambiente BrainTumorRL"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if args.continuous:
        # I trackbar di OpenCV lavorano solo con interi positivi.
        # Definiamo un range 0-200, dove il valore di default 100 mappa a 0.0.
        def nothing(x): pass
        cv2.createTrackbar("dx (Centro X)", window_name, 100, 200, nothing)
        cv2.createTrackbar("dy (Centro Y)", window_name, 100, 200, nothing)
        cv2.createTrackbar("dw (Larghezza)", window_name, 100, 200, nothing)
        cv2.createTrackbar("dh (Altezza)", window_name, 100, 200, nothing)

    obs, _ = env.reset()
    done = False
    last_info = None

    while True:
        # Genera il frame standard dell'ambiente
        frame = env.render()
        
        # Costruiamo una sezione inferiore extra (HUD) nera per stampare la scomposizione dettagliata del reward
        h, w, _ = frame.shape
        hud_panel = np.zeros((110, w, 3), dtype=np.uint8)
        
        # Titolo della modalità corrente
        mode_text = "MOD: SAC (Continuo) | Regola slider + SPAZIO per confermare" if args.continuous else "MOD: PPO (Discreto) | Usa W/A/S/D, Q/E, F/R | SPAZIO=STOP"
        cv2.putText(hud_panel, mode_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        
        if last_info and "rew_components" in last_info:
            rc = last_info["rew_components"]
            rew_line = f"Reward Step: {rc.get('total', 0.0):+.3f}  [dIoU: {rc.get('delta_iou', 0.0):+.3f} | dDist: {rc.get('delta_dist', 0.0):+.3f}]"
            pen_line = f"Penalita':  [Oversize: {rc.get('oversize_penalty', 0.0):+.3f} | Osc/Smooth: {rc.get('oscillation_penalty', 0.0):+.3f}]"
            
            cv2.putText(hud_panel, rew_line, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(hud_panel, pen_line, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 140, 255), 1, cv2.LINE_AA)
            
            if done:
                cv2.putText(hud_panel, "EPISODIO TERMINATO! Premi 'N' per ricominciare.", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(hud_panel, "In attesa della prima mossa per calcolare il reward...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

        # Uniamo verticalmente il render dell'ambiente e il nostro HUD personalizzato
        full_display = np.vstack([frame, hud_panel])
        cv2.imshow(window_name, full_display)

        # Aspetta l'input da tastiera (timeout di 30ms per tenere fluida la UI)
        key = cv2.waitKey(30) & 0xFF

        # Esci dalla simulazione (ESC o Q)
        if key == 27 or key == ord('q') or key == ord('Q'):
            print("[SIMULAZIONE] Sessione conclusa dall'utente.")
            break
            
        # Forza il reset / Cambio immagine (N)
        if key == ord('n') or key == ord('N'):
            print("[SIMULAZIONE] Reset dell'ambiente caricato su un nuovo campione casuale.")
            obs, _ = env.reset()
            done = False
            last_info = None
            if args.continuous:
                cv2.setTrackbarPos("dx (Centro X)", window_name, 100)
                cv2.setTrackbarPos("dy (Centro Y)", window_name, 100)
                cv2.setTrackbarPos("dw (Larghezza)", window_name, 100)
                cv2.setTrackbarPos("dh (Altezza)", window_name, 100)
            continue

        # Se l'episodio è finito (Truncated o Terminated), blocca gli step finché l'utente non resetta con 'N'
        if done:
            continue

        action = None
        
        # Logica di cattura azione se l'ambiente è DISCRETO (PPO)
        if not args.continuous:
            # Mappatura azioni: 0: left, 1: right, 2: up, 3: down, 4: w+, 5: w-, 6: h+, 7: h-, 8: STOP
            if key == ord('a') or key == ord('A'):   action = 0
            elif key == ord('d') or key == ord('D'): action = 1
            elif key == ord('w') or key == ord('W'): action = 2
            elif key == ord('s') or key == ord('S'): action = 3
            elif key == ord('e') or key == ord('E'): action = 4
            elif key == ord('q') or key == ord('Q'): action = 5
            elif key == ord('r') or key == ord('R'): action = 6
            elif key == ord('f') or key == ord('F'): action = 7
            elif key == 32:  # Barra Spazio -> STOP
                action = 8
                
        # Logica di cattura azione se l'ambiente è CONTINUO (SAC)
        else:
            if key == 32:  # Barra Spazio -> Applica i trackbar correnti
                # Estrai le posizioni degli slider (intervallo 0-200)
                tx = cv2.getTrackbarPos("dx (Centro X)", window_name)
                ty = cv2.getTrackbarPos("dy (Centro Y)", window_name)
                tw = cv2.getTrackbarPos("dw (Larghezza)", window_name)
                th = cv2.getTrackbarPos("dh (Altezza)", window_name)
                
                # Sottrai 100 e dividi per 100 per rimappare linearmente in [-1.0, 1.0]
                dx = (tx - 100) / 100.0
                dy = (ty - 100) / 100.0
                dw = (tw - 100) / 100.0
                dh = (th - 100) / 100.0
                
                action = np.array([dx, dy, dw, dh], dtype=np.float32)

        # Se l'utente ha impartito un comando valido, esegui lo step nell'ambiente
        if action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            last_info = info
            done = terminated or truncated
            
            if done:
                print(f"[SIMULAZIONE] Episodio concluso! Passi totali: {env.current_step}. IoU Finale: {info.get('iou_instant', 0.0):.4f}")
                if "episode_metrics" in info:
                    print(f"               Metrica IoU Media Episodio: {info['episode_metrics'].get('ep_iou_mean', 0.0):.4f}")

    cv2.destroyAllWindows()


def preview_datasets_as_video(train_ds, test_ds):
    import cv2
    import numpy as np
    
    print("\n" + "="*70)
    print("[PREVIEW] ATTIVAZIONE ANTEPRIMA VISIVA PRE-TRAINING")
    print("[PREVIEW] Le immagini scorreranno automaticamente a schermo.")
    print("[PREVIEW] - Premi 'SPAZIO' per mettere in PAUSA / RIPRENDERE")
    print("[PREVIEW] - Premi 'Q' in qualunque momento per iniziare il TRAINING.")
    print("="*70 + "\n")
    
    cv2.namedWindow("Verifica Preprocessing (Scala di Grigi)", cv2.WINDOW_NORMAL)
    
    paused = False
    for name, ds in [("TRAIN SET", train_ds), ("TEST SET", test_ds)]:
        print(f"[PREVIEW] Avvio riproduzione {name} ({len(ds)} immagini totali)...")
        i = 0
        while i < len(ds):
            sample = ds[i]
            img_tensor = sample["image"].numpy()     # Shape: [1, 224, 224]
            mask_tensor = sample["mask"].numpy().squeeze(0) # Shape: [224, 224]
            
            img_gray = img_tensor[0]
            if img_gray.max() <= 1.0:
                img_gray = (img_gray * 255).astype(np.uint8)
            else:
                img_gray = img_gray.astype(np.uint8)
                
            display_frame = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            
            contours, _ = cv2.findContours((mask_tensor * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display_frame, contours, -1, (0, 255, 0), 2)
            
            cv2.putText(display_frame, f"{name} - Idx: {i}/{len(ds)}", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, f"Polarity: {sample['polarity']}", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            
            cv2.imshow("Verifica Preprocessing (Scala di Grigi)", display_frame)
            
            delay = 0 if paused else 100
            key = cv2.waitKey(delay) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("[PREVIEW] Anteprima interrotta dall'utente. Avvio del training in corso...")
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):  
                paused = not paused
                print(f"[PREVIEW] Stato di pausa: {paused}")
                continue
                
            if not paused:
                i += 1
                
    cv2.destroyAllWindows()
    print("[PREVIEW] Visualizzazione completata di tutti i campioni.")

def main():
    args = build_arg_parser().parse_args()
    cfg = build_dataset_config(args)
    os.makedirs(cfg["output"]["root"], exist_ok=True)

    print(f"[data] Sorgente dataset selezionata: '{cfg['dataset']['source']}'")
    train_ds, val_ds, test_ds = get_datasets(cfg)

    # --- INSERISCI LA CHIAMATA QUI SOTTO ---
    #preview_datasets_as_video(train_ds, test_ds)
    # ---------------------------------------
    
    # ── NUOVA MODALITÀ: Simulazione interattiva manuale ed immediata ────
    if args.simulate:
        run_manual_simulation(train_ds, args)
        return

    # ── warm-start supervisionato (vedi localizer.py) ───────────────────
    localizer_path = args.localizer_path or os.path.join(cfg["output"]["root"], "localizer.pt")
    if args.train_localizer:
        print(f"[localizer] training del regressore supervisionato ({args.localizer_epochs} epoche)...")
        train_localizer(train_ds, val_ds, save_path=localizer_path, epochs=args.localizer_epochs)

    localizer_fn = None
    if args.use_localizer:
        if not os.path.exists(localizer_path):
            raise SystemExit(f"--use-localizer richiesto ma nessun checkpoint trovato in {localizer_path}. "
                              f"Usa anche --train-localizer, oppure passa --localizer-path corretto.")
        loc_model, loc_device = load_localizer(localizer_path, device="cpu")
        localizer_fn = make_localizer_fn(loc_model, loc_device)
        print(f"[localizer] warm-start attivo, checkpoint: {localizer_path}")

    # ── modalità: solo valutazione visiva su un modello già allenato ────
    if args.eval_only or args.watch_idx is not None:
        model, _ = load_model_and_vecenv(args, make_eval_env_fn(val_ds, localizer_fn=localizer_fn, continuous_actions=args.continuous))
        evaluator = VisualEvaluator(model=model, test_ds=test_ds,
                                     output_dir=os.path.join(cfg["output"]["root"], "test_eval"),
                                     max_steps=MAX_STEPS_PER_EPISODE, seed=cfg["seed"],
                                     localizer_fn=localizer_fn, continuous_actions=args.continuous)
        if args.watch_idx is not None:
            evaluator.watch(args.watch_idx)
            evaluator.save_episode_video(args.watch_idx)
            evaluator.save_episode_steps_grid(args.watch_idx)
        else:
            evaluator.evaluate_dataset(max_samples=0)
        return

    # ── training ──────────────────────────────────────────────────────
    check_env(BrainTumorRL_Env(pytorch_dataset=train_ds, max_steps=MAX_STEPS_PER_EPISODE,
                                localizer_fn=localizer_fn, continuous_actions=args.continuous))

    N_ENVS = args.n_envs
    N_EPOCHS = args.n_epochs
    GAMMA = 0.99
    N_STEPS = 1024
    STOP_CURRICULUM_STEPS = 350_000

    INITIAL_MIN_STEPS_BEFORE_STOP = MAX_STEPS_PER_EPISODE + 1
    FINAL_MIN_STEPS_BEFORE_STOP = 0

    env_fn = make_train_env_fn(train_ds, INITIAL_MIN_STEPS_BEFORE_STOP, localizer_fn=localizer_fn,
                                continuous_actions=args.continuous)
    vec_env = SubprocVecEnv([env_fn for _ in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=GAMMA)

    eval_env_raw = DummyVecEnv([make_eval_env_fn(val_ds, localizer_fn=localizer_fn, continuous_actions=args.continuous)])
    eval_env_raw = VecMonitor(eval_env_raw)
    eval_env = VecNormalize(eval_env_raw, norm_obs=False, norm_reward=False, gamma=GAMMA, training=False)

    visual_callback = VisionMetricsCallback(val_dataset=val_ds,
                                             save_dir=os.path.join(cfg["output"]["root"], "gradcam_outputs"),
                                             continuous=args.continuous)
    adaptive_curriculum_callback = AdaptiveCurriculumCallback(
        n_stages=6, initial_difficulty=0.0, final_difficulty=1.0,
        window=50, min_steps_per_stage=5_000, advance_threshold=0.55,
        regress_threshold=0.20, stall_patience=300_000,
        reheat_ent=0.03, floor_ent=0.01,
        reheat_step_frac=0.05, floor_step_frac=0.012,
        reheat_duration=60_000,
        manage_entropy=not args.continuous,
    )

    callback_list_items = [visual_callback, adaptive_curriculum_callback]
    if not args.continuous:
        stop_curriculum_callback = StopCurriculumCallback(curriculum_steps=STOP_CURRICULUM_STEPS,
                                                           initial_min_steps=INITIAL_MIN_STEPS_BEFORE_STOP,
                                                           final_min_steps=FINAL_MIN_STEPS_BEFORE_STOP)
        callback_list_items.append(stop_curriculum_callback)

    checkpoint_dir = os.path.join(cfg["output"]["root"], "checkpoints")
    model_checkpoint_callback = ModelCheckpointCallback(
        save_freq=args.checkpoint_every, save_dir=checkpoint_dir, vec_env=vec_env,
        name_prefix="ppo_brain_tumor", keep_last=10,
    )

    stop_on_no_improve = StopTrainingOnNoModelImprovement(max_no_improvement_evals=120, min_evals=180, verbose=1)
    if args.continuous:
        eval_callback = EvalCallback(
            eval_env, best_model_save_path=os.path.join(cfg["output"]["root"], "best_model/"),
            log_path=os.path.join(cfg["output"]["root"], "eval_results/"),
            eval_freq=max(N_STEPS * 4, 2000), n_eval_episodes=20, deterministic=True,
            callback_after_eval=stop_on_no_improve, verbose=1,
        )
    else:
        eval_callback = MaskableEvalCallback(
            eval_env, best_model_save_path=os.path.join(cfg["output"]["root"], "best_model/"),
            log_path=os.path.join(cfg["output"]["root"], "eval_results/"),
            eval_freq=max(N_STEPS * 4, 2000), n_eval_episodes=20, deterministic=True,
            use_masking=True, callback_after_eval=stop_on_no_improve, verbose=1,
        )
    callback_list_items += [model_checkpoint_callback, eval_callback]
    callbacks = CallbackList(callback_list_items)

    if args.total_timesteps is not None:
        total_timesteps = args.total_timesteps
    else:
        total_timesteps = int(os.environ.get("TOTAL_TIMESTEPS", 100_000))

    if args.continuous:
        policy_kwargs = dict(features_extractor_kwargs=dict(cnn_output_dim=512),
                              net_arch=dict(pi=[256, 256], qf=[256, 256]))
        model = SAC(
            policy="MultiInputPolicy", env=vec_env, policy_kwargs=policy_kwargs,
            learning_rate=3e-4, buffer_size=args.sac_buffer_size,
            learning_starts=args.sac_learning_starts, batch_size=256,
            tau=0.005, gamma=GAMMA, train_freq=1, gradient_steps=1,
            ent_coef="auto", optimize_memory_usage=False,
            replay_buffer_kwargs=dict(handle_timeout_termination=False),
            verbose=1, tensorboard_log=os.path.join(cfg["output"]["root"], "tb"),
        )
    else:
        policy_kwargs = dict(features_extractor_kwargs=dict(cnn_output_dim=512),
                              net_arch=dict(pi=[256, 256], vf=[256, 256]))
        model = MaskablePPO(
            policy="MultiInputPolicy", env=vec_env, policy_kwargs=policy_kwargs,
            learning_rate=linear_schedule(1.5e-4, 1e-5), n_steps=N_STEPS, batch_size=1024,
            n_epochs=N_EPOCHS, gamma=GAMMA, gae_lambda=0.95, clip_range=linear_schedule(0.1, 0.03),
            ent_coef=0.03, vf_coef=0.5, max_grad_norm=0.5, verbose=1,
            tensorboard_log=os.path.join(cfg["output"]["root"], "tb"), target_kl=0.03,
        )

    for i in range(1, args.n_iterations + 1):
        model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=(i == 1))
        model.save(os.path.join(cfg["output"]["root"], f"{total_timesteps * i}"))

    vec_env.save(os.path.join(cfg["output"]["root"], "vecnormalize_stats.pkl"))

    eval_output_dir = os.path.join(cfg["output"]["root"], "test_eval")
    evaluator = VisualEvaluator(model=model, test_ds=test_ds,
                                 output_dir=eval_output_dir,
                                 max_steps=MAX_STEPS_PER_EPISODE, seed=cfg["seed"],
                                 localizer_fn=localizer_fn, continuous_actions=args.continuous)
    evaluator.evaluate_dataset(max_samples=0)


if __name__ == "__main__":
    main()