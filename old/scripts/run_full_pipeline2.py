#!/usr/bin/env python3
"""
run_full_pipeline_2.py
Pipeline: 
  1) Load dataset
  2) Train U-Net
  3) Evaluate Baseline (Full Image)
  4) Train RL Agent (as ROI Finder)
  5) Evaluate Pipeline (RL ROI Finder + U-Net on Crop)
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Import locali
from src.utils.io_utils import load_config, set_seed, get_device, ensure_dirs
from src.data.dataset import get_datasets, make_dataloaders
from src.models.unet import build_unet, build_loss
from src.train.trainer import BaselineTrainer, build_optimizer_scheduler
from src.train.rl_trainer import RLTrainer
from src.rl.environment2 import ROIFinderEnv # Importa la nuova classe ambiente
from src.rl.agent import DQNAgent
from src.eval.metrics import compute_metrics, average_metrics

def main():
    # Setup base
    cfg = load_config(os.path.join(os.path.dirname(__file__), "../configs", "config.yaml"))
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    ensure_dirs(cfg)

    # 1) Dataset
    train_ds, val_ds, test_ds = get_datasets(cfg)
    train_loader, _, test_loader = make_dataloaders(cfg, train_ds, val_ds, test_ds)

    # 2) U-Net Baseline
    model = build_unet(cfg).to(device)
    trainer = BaselineTrainer(model, build_loss(cfg), *build_optimizer_scheduler(model, cfg), device, cfg)
    trainer.train(train_loader, None, epochs=cfg["training"]["epochs"])
    trainer.load_best()

    # 3) Train RL Agent (ROI Finder)
    print("\n[pipeline] Step 5: Training RL for ROI localization...")
    # L'ambiente ora inizializza il Bounding Box per trovare il tumore
    env = ROIFinderEnv(cfg)
    agent = DQNAgent(cfg, device)
    rl_trainer = RLTrainer(model, agent, env, train_loader, None, device, cfg)
    rl_trainer.train()

    # 4) Evaluate Pipeline: RL + U-Net on Crop
    print("\n[pipeline] Step 6: Evaluating ROI-based RL + U-Net...")
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            targets = batch["mask"]
            
            for i in range(imgs.size(0)):
                img_np = imgs[i].cpu().numpy().squeeze()
                gt_np = targets[i].numpy().squeeze()
                
                # Agente RL trova la ROI
                state = env.reset(img_np, gt_np)
                done = False
                while not done:
                    action = agent.act(state, greedy=True)
                    state, _, done, _ = env.step(action)
                
                # Crop e Resize dinamico
                x1, y1, x2, y2 = map(int, env.box)
                crop = imgs[i, :, y1:y2, x1:x2].unsqueeze(0)
                crop_r = F.interpolate(crop, size=(cfg['dataset']['image_size'], cfg['dataset']['image_size']))
                
                # U-Net segmenta solo la ROI[cite: 11]
                logits = model(crop_r)
                pred = (torch.sigmoid(logits) > 0.5).float().cpu().numpy().squeeze()
                
                # Calcolo metriche
                all_metrics.append(compute_metrics(pred, gt_np))
    
    avg_results = average_metrics(all_metrics)
    print(f"\n[pipeline] Final Metrics with RL ROI Finder: {avg_results}")
    
    # Salvataggio risultati
    pd.DataFrame([avg_results]).to_csv(os.path.join(cfg["output"]["metrics_dir"], "final_results.csv"))
    print("\nPipeline complete!")

if __name__ == "__main__":
    main()