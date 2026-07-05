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
"""
import argparse
import os

from stable_baselines3.common.callbacks import (
    CallbackList,
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
            "image_size": [224, 224], "in_channels": 3,
            "train_ratio": 0.8, "val_ratio": 0.1, "cache_pairs": False,
        },
        "preprocessing": {"normalization": "minmax", "binarize_mask": True, "mask_threshold": 0.5},
        "training": {"batch_size": 512, "num_workers": 0},
        "output": {"root": args.output_root},
        "seed": 42,
    }


def make_train_env_fn(train_ds, initial_min_steps_before_stop):
    def _init():
        env = BrainTumorRL_Env(
            pytorch_dataset=train_ds, max_steps=MAX_STEPS_PER_EPISODE,
            min_steps_before_stop=initial_min_steps_before_stop, init_difficulty=0.0,
        )
        return ActionMasker(env, mask_fn)
    return _init


def make_eval_env_fn(val_ds, init_difficulty=0.5):
    # FIX: prima non veniva passato init_difficulty, quindi l'eval-env usava
    # il default della classe (1.0, il piu' difficile) fin dall'inizio del
    # training, mentre il train-env parte a difficolta' 0. Questo disallineamento
    # rendeva la valutazione artificiosamente "stagnante" nelle prime centinaia
    # di migliaia di step, rischiando di far scattare lo StopTrainingOnNoModelImprovement
    # per il motivo sbagliato. Si valuta a difficolta' intermedia di default;
    # la valutazione finale su test set (fuori da questo callback) resta comunque
    # sull'intero test set reale.
    def _init():
        env = BrainTumorRL_Env(pytorch_dataset=val_ds, max_steps=MAX_STEPS_PER_EPISODE,
                                min_steps_before_stop=0, init_difficulty=init_difficulty)
        return ActionMasker(env, mask_fn)
    return _init


def load_model_and_vecenv(args, dummy_env_fn):
    """Carica modello + VecNormalize da un checkpoint per eval-only / watch."""
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

    model = MaskablePPO.load(args.model_path)
    return model, vec_env


def main():
    args = build_arg_parser().parse_args()
    cfg = build_dataset_config(args)
    os.makedirs(cfg["output"]["root"], exist_ok=True)

    print(f"[data] Sorgente dataset selezionata: '{cfg['dataset']['source']}'")
    train_ds, val_ds, test_ds = get_datasets(cfg)

    # ── modalità: solo valutazione visiva su un modello già allenato ────
    if args.eval_only or args.watch_idx is not None:
        model, _ = load_model_and_vecenv(args, make_eval_env_fn(val_ds))
        evaluator = VisualEvaluator(model=model, test_ds=test_ds,
                                     output_dir=os.path.join(cfg["output"]["root"], "test_eval"),
                                     max_steps=MAX_STEPS_PER_EPISODE, seed=cfg["seed"])
        if args.watch_idx is not None:
            evaluator.watch(args.watch_idx)
            evaluator.save_episode_video(args.watch_idx)
            evaluator.save_episode_steps_grid(args.watch_idx)
        else:
            evaluator.evaluate_dataset(max_samples=0)
        return

    # ── training ──────────────────────────────────────────────────────
    check_env(BrainTumorRL_Env(pytorch_dataset=train_ds, max_steps=MAX_STEPS_PER_EPISODE))

    N_ENVS = args.n_envs
    N_EPOCHS = args.n_epochs
    GAMMA = 0.99
    # FIX v2: N_STEPS=512 dava un rollout buffer piccolo (512*6 env=3072), stime
    # del value/vantaggio piu' rumorose (vedi train/explained_variance oscillare
    # parecchio invece di crescere stabile). Un buffer piu' grande media meglio
    # il rumore ad ogni update.
    N_STEPS = 1024

    # FIX v4: rimosso il curriculum a step_count fisso (v3), sostituito da uno
    # auto-paced gated dalle performance (vedi AdaptiveCurriculumCallback in
    # callbacks.py). STOP_CURRICULUM_STEPS resta a tempo fisso perche' "quando
    # posso iniziare a valutare se fermarmi" e' una competenza generale, non
    # legata allo stage di difficolta' attuale.
    STOP_CURRICULUM_STEPS = 350_000

    INITIAL_MIN_STEPS_BEFORE_STOP = MAX_STEPS_PER_EPISODE + 1
    FINAL_MIN_STEPS_BEFORE_STOP = 0

    env_fn = make_train_env_fn(train_ds, INITIAL_MIN_STEPS_BEFORE_STOP)
    vec_env = SubprocVecEnv([env_fn for _ in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=GAMMA)

    eval_env_raw = DummyVecEnv([make_eval_env_fn(val_ds)])
    eval_env_raw = VecMonitor(eval_env_raw)
    eval_env = VecNormalize(eval_env_raw, norm_obs=False, norm_reward=False, gamma=GAMMA, training=False)

    visual_callback = VisionMetricsCallback(val_dataset=val_ds,
                                             save_dir=os.path.join(cfg["output"]["root"], "gradcam_outputs"))
    # FIX v4: un solo callback auto-paced gestisce difficolta' + reheat di
    # ent_coef/step_frac (vedi AdaptiveCurriculumCallback in callbacks.py per il
    # perche'). Avanza solo quando le performance lo permettono e regredisce se
    # crollano, invece di seguire un contatore di step cieco.
    adaptive_curriculum_callback = AdaptiveCurriculumCallback(
        n_stages=6, initial_difficulty=0.0, final_difficulty=1.0,
        window=100, min_steps_per_stage=30_000, advance_threshold=0.35,
        regress_threshold=0.08, stall_patience=150_000,
        reheat_ent=0.03, floor_ent=0.01,
        reheat_step_frac=0.05, floor_step_frac=0.012,
        reheat_duration=60_000,
    )
    stop_curriculum_callback = StopCurriculumCallback(curriculum_steps=STOP_CURRICULUM_STEPS,
                                                       initial_min_steps=INITIAL_MIN_STEPS_BEFORE_STOP,
                                                       final_min_steps=FINAL_MIN_STEPS_BEFORE_STOP)

    checkpoint_dir = os.path.join(cfg["output"]["root"], "checkpoints")
    model_checkpoint_callback = ModelCheckpointCallback(
        save_freq=args.checkpoint_every, save_dir=checkpoint_dir, vec_env=vec_env,
        name_prefix="ppo_brain_tumor", keep_last=10,
    )

    # FIX: eval_env valuta sempre a init_difficulty=1.0 (default della classe,
    # vedi make_eval_env_fn) mentre il train-env parte a difficolta' 0 e sale
    # gradualmente: per gran parte del training la valutazione sembra "piatta"
    # semplicemente perche' il compito e' piu' difficile di quanto l'agente
    # abbia ancora praticato, non perche' l'agente abbia smesso di migliorare.
    # Con pazienza troppo bassa questo fa scattare lo stop anticipato proprio
    # mentre il curriculum di difficolta' e' ancora a meta' corsa (e' quello
    # che e' successo intorno a 1.1M step nella run analizzata). Si alza la
    # pazienza per dare tempo al curriculum di completarsi.
    stop_on_no_improve = StopTrainingOnNoModelImprovement(max_no_improvement_evals=120, min_evals=180, verbose=1)
    eval_callback = MaskableEvalCallback(
        eval_env, best_model_save_path=os.path.join(cfg["output"]["root"], "best_model/"),
        log_path=os.path.join(cfg["output"]["root"], "eval_results/"),
        eval_freq=max(N_STEPS * 4, 2000), n_eval_episodes=20, deterministic=True,
        use_masking=True, callback_after_eval=stop_on_no_improve, verbose=1,
    )

    callbacks = CallbackList([
        visual_callback, adaptive_curriculum_callback, stop_curriculum_callback,
        model_checkpoint_callback, eval_callback,
    ])

    policy_kwargs = dict(features_extractor_kwargs=dict(features_dim=512),
                          net_arch=dict(pi=[256, 256], vf=[256, 256]))

    if args.total_timesteps is not None:
        total_timesteps = args.total_timesteps
    else:
        total_timesteps = int(os.environ.get("TOTAL_TIMESTEPS", 100_000))

    model = MaskablePPO(
        policy="CnnPolicy", env=vec_env, policy_kwargs=policy_kwargs,
        learning_rate=linear_schedule(1.5e-4, 1e-5), n_steps=N_STEPS, batch_size=1024,
        n_epochs=N_EPOCHS, gamma=GAMMA, gae_lambda=0.95, clip_range=linear_schedule(0.1, 0.03),
        # FIX v3: ent_coef allineato al reheat_ent di StagedCurriculumCallback (0.03),
        # che e' il valore con cui parte ogni stage (compreso il primo). vf_coef
        # resta a 0.5 (standard PPO) invece dell'1.0 originale, per non far
        # dominare la value loss sul gradiente di policy.
        ent_coef=0.03, vf_coef=0.5, max_grad_norm=0.5, verbose=1,
        tensorboard_log=os.path.join(cfg["output"]["root"], "tb"), target_kl=0.03,
    )

    # FIX: reset_num_timesteps=False sempre (anche alla prima chiamata) faceva si'
    # che SB3 riusasse la stessa cartella tensorboard ("MaskablePPO_0") ad ogni
    # riavvio dello script, invece di crearne una nuova incrementale (_1, _2, ...).
    # reset_num_timesteps deve essere True solo alla primissima iterazione del run.
    for i in range(1, args.n_iterations + 1):
        model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=(i == 1))
        model.save(os.path.join(cfg["output"]["root"], f"{total_timesteps * i}"))

    vec_env.save(os.path.join(cfg["output"]["root"], "vecnormalize_stats.pkl"))

    # ── valutazione visiva finale sul test set ───────────────────────────
    evaluator = VisualEvaluator(model=model, test_ds=test_ds,
                                 output_dir=os.path.join(cfg["output"]["root"], "test_eval"),
                                 max_steps=MAX_STEPS_PER_EPISODE, seed=cfg["seed"])
    evaluator.evaluate_dataset(max_samples=0)


if __name__ == "__main__":
    main()
