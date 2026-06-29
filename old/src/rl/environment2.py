import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ROIFinderEnv(gym.Env):
    def __init__(self, cfg):
        super(ROIFinderEnv, self).__init__()
        self.img_size = cfg['dataset']['image_size']
        self.step_size = int(self.img_size * 0.05)
        # Azioni: 0:Su, 1:Giù, 2:Sinistra, 3:Destra, 4:Allarga, 5:Stringi, 6:Stop
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, self.img_size, self.img_size), dtype=np.float32)

    def reset(self, image=None, gt_mask=None):
        self.image = image
        self.gt_mask = gt_mask
        # Calcola GT box
        y, x = np.where(self.gt_mask > 0.5)
        self.gt_box = [np.min(x), np.min(y), np.max(x), np.max(y)] if len(y) > 0 else [0,0,self.img_size,self.img_size]
        # Box iniziale centrale
        m = self.img_size // 4
        self.box = [m, m, self.img_size - m, self.img_size - m]
        self.best_iou = self._compute_iou(self.box, self.gt_box)
        return self._get_state()

    def step(self, action):
        x1, y1, x2, y2 = self.box
        if action == 0: y1 -= self.step_size; y2 -= self.step_size
        elif action == 1: y1 += self.step_size; y2 += self.step_size
        elif action == 2: x1 -= self.step_size; x2 -= self.step_size
        elif action == 3: x1 += self.step_size; x2 += self.step_size
        elif action == 4: x1-=self.step_size; y1-=self.step_size; x2+=self.step_size; y2+=self.step_size
        elif action == 5: x1+=self.step_size; y1+=self.step_size; x2-=self.step_size; y2-=self.step_size
        
        self.box = [np.clip(x1, 0, self.img_size), np.clip(y1, 0, self.img_size), 
                    np.clip(x2, x1+10, self.img_size), np.clip(y2, y1+10, self.img_size)]
        
        new_iou = self._compute_iou(self.box, self.gt_box)
        reward = (new_iou - self.best_iou) * 100
        self.best_iou = max(new_iou, self.best_iou)
        return self._get_state(), reward, (action == 6), {}

    def _get_state(self):
        state = np.zeros((2, self.img_size, self.img_size), dtype=np.float32)
        state[0] = self.image
        x1, y1, x2, y2 = map(int, self.box)
        state[1, y1:y2, x1:x2] = 1.0
        return state

    def _compute_iou(self, bA, bB):
        xA, yA = max(bA[0], bB[0]), max(bA[1], bB[1])
        xB, yB = min(bA[2], bB[2]), min(bA[3], bB[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        return inter / float((bA[2]-bA[0])*(bA[3]-bA[1]) + (bB[2]-bB[0])*(bB[3]-bB[1]) - inter + 1e-6)