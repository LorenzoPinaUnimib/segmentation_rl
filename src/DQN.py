import cv2, gymnasium, json, kagglehub, numpy as np, os, random, tensorflow as tf
from collections import deque
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class BrainTumorEnv(gymnasium.Env):
    def __init__(self, images, target_boxes):
        super(BrainTumorEnv, self).__init__()

        # Imposta immagini e target_boxes
        self.images = images
        self.target_boxes = target_boxes

        # Imposta immagine attuale alla 0
        self.current_img_idx = 0

        # Definisce il numero di azioni disponibili
        self.action_space = spaces.Discrete(8)

        # Definizione dell'immagine e del box
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(640, 640, 3), dtype=np.uint8),
            "box": spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        })

        # Imposta il conteggio attuale e il numero massimo di passi per episodio
        self.step_count = 0
        self.max_steps = 200
        # Variabili per il reward shaping
        self.prev_dist = None
        self.prev_iou = None

    def reset(self, seed=None):
        super().reset(seed=seed)

        # Sceglie un'immagine casuale
        self.current_img_idx = random.randint(0, len(self.images) - 1)

        # Definiamo una box iniziale al centro dell'immagine e di dimensione pari al 100% della stessa
        # TODO: diminuire dimensione box
        box_size = 1.0
        start_x = 0.5 - (box_size / 2)
        start_y = 0.5 - (box_size / 2)
        
        self.agent_box = np.array([start_x, start_y, box_size, box_size], dtype=np.float32)
        
        # Imposta il numero di passi a 0
        self.step_count = 0
        # Resetta i valori precedenti per il reward shaping
        self.prev_dist = None
        self.prev_iou = None

        # Restituisce l'immagine e la box
        observation = {
            "image": self.images[self.current_img_idx], 
            "box": self.agent_box
        }

        return observation, {}

    def step(self, action):
        # Incrementiamo il numero di passi
        self.step_count += 1

        # Definiamo la dimensione del movimento
        move_size = 0.01

        # Determino la modifica da apportare alla box in base all'azione scelta
        if action == 0: self.agent_box[1] -= move_size 
        elif action == 1: self.agent_box[1] += move_size 
        elif action == 2: self.agent_box[0] -= move_size
        elif action == 3: self.agent_box[0] += move_size
        elif action == 4:
            self.agent_box[0] -= move_size / 2
            self.agent_box[2] += move_size
        elif action == 5:
            self.agent_box[0] += move_size / 2
            self.agent_box[2] -= move_size
        elif action == 6:
            self.agent_box[1] -= move_size / 2
            self.agent_box[3] += move_size
        elif action == 7:
            self.agent_box[1] += move_size / 2
            self.agent_box[3] -= move_size

        # Impedisco alla box di uscire dall'immagine
        self.agent_box = np.clip(self.agent_box, 0, 1)

        # Calcola la ricompensa
        reward = self.calculate_reward()

        # Recupero la box target dell'immagine
        target = self.target_boxes[self.current_img_idx]

        # Calcolo la IoU e termino se il valore è sopra a 0.7
        terminated = False

        if self.calculate_iou(self.agent_box, target) > 0.7:
            terminated = True
            if self.calculate_iou(self.agent_box, target) > 0.9:
                reward += 100
            else:
                reward += 40

        # Controllo se ho raggiunto il massimo numero di passi
        truncated = False

        if self.step_count >= self.max_steps:
            truncated = True

        # Restituisco l'immagine, la box, la reward totale e le informazioni sulla terminazione
        observation = {
            "image": self.images[self.current_img_idx], 
            "box": self.agent_box
        }

        return observation, reward, terminated, truncated, {}

    def calculate_iou(self, boxA, boxB):
        # Calcolo la Intersection over Union
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    # TODO: calcolare la reward in base al passo precedente
    def calculate_reward(self):
        # Calcolo la IoU sulla box determinata dall'agente e quella target
        target = self.target_boxes[self.current_img_idx]
        iou = self.calculate_iou(self.agent_box, target)

        # Definisco reward base per questo passo
        reward = 0.0

        # Aumento la reward in base all'avvicinamento dell'agente alla dimensione corretta
        if self.prev_iou is not None:
            iou_improvement = iou - self.prev_iou

            if iou_improvement > 0:
                reward += iou_improvement * 100.0
            else:
                reward += iou_improvement * 110.0
        
        # Salvo la IoU attuale
        self.prev_iou = iou

        # Calcolo il centro delle due box
        center_agent = np.array([self.agent_box[0] + self.agent_box[2]/2, self.agent_box[1] + self.agent_box[3]/2])
        center_target = np.array([target[0] + target[2]/2, target[1] + target[3]/2])

        # Determino la distanza tra i due centri
        dist = np.linalg.norm(center_agent - center_target)

        # Aumento la reward in base all'avvicinamento delle due box
        if self.prev_dist is not None:
            dist_improvement = self.prev_dist - dist

            if (dist_improvement > 0):
                reward += dist_improvement * 50.0
            else:
                reward += dist_improvement * 60.0
        
        # Salvo la distanza attuale
        self.prev_dist = dist

        # Penalizzazione per ogni passo aggiuntivo
        reward -= 0.01
        
        return reward

def build_dqn_model():
    # Prendo in input l'immagine
    img_input = tf.keras.layers.Input(shape=(640, 640, 3), name="image")

    # Riduco la dimensione dell'immaigne e normalizzo i pixel
    x = tf.keras.layers.Resizing(224, 224)(img_input)
    x = tf.keras.layers.Rescaling(1./255)(x) 

    # Estrazione delle feature dall'immagine ridotta
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)

    # Prendo in input la box attuale
    box_input = tf.keras.layers.Input(shape=(4,), name="box")
    x2 = tf.keras.layers.Dense(32, activation='relu')(box_input)

    # Concateno l'immagine e la box
    combined = tf.keras.layers.Concatenate()([x, x2])

    # Uso una rete che produce in output 8 valori, uno per ogni azione
    dense = tf.keras.layers.Dense(128, activation='relu')(combined)
    dense = tf.keras.layers.Dense(64, activation='relu')(dense)
    output = tf.keras.layers.Dense(8, activation='linear')(dense)

    # Uso Adam come ottimizzatore
    # TODO: variare LR e vedere performance
    model = tf.keras.Model(inputs=[img_input, box_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007), loss='mse')
    return model

class DQNAgent:
    def __init__(self, action_size):
        self.action_size = action_size

        # Definisce la dimensione della memoria delle ultime esperienze
        # TODO: testare aumento della memoria
        self.memory = deque(maxlen=500) 

        # Quanto conta il futuro rispetto ad adesso
        self.gamma = 0.95

        # Fattore di esplorazione
        self.epsilon = 1.0

        # Fattore minimo di esplorazione
        self.epsilon_min = 0.01

        # Diminuzione dopo ogni allenamento (viene moltiplicato ogni volta)
        self.epsilon_decay = 0.9975

        # Crea la rete neurale
        self.model = build_dqn_model()
        self.target_model = build_dqn_model()
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, training=True):
        # Sceglie se esplorare o meno
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Scelgo l'azione che ha valore massimo in base al modello
        img = np.expand_dims(state['image'], axis=0)
        box = np.expand_dims(state['box'], axis=0)
        act_values = self.model.predict([img, box], verbose=0)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        # Salvo un'esperienza
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        # Estraggo un minibatch
        minibatch = random.sample(self.memory, batch_size)

        states_img = np.array([s['image'] for s, a, r, ns, d in minibatch])
        states_box = np.array([s['box'] for s, a, r, ns, d in minibatch])
        next_states_img = np.array([ns['image'] for s, a, r, ns, d in minibatch])
        next_states_box = np.array([ns['box'] for s, a, r, ns, d in minibatch])
        rewards = np.array([r for s, a, r, ns, d in minibatch])
        actions = np.array([a for s, a, r, ns, d in minibatch])
        dones = np.array([d for s, a, r, ns, d in minibatch])

        targets = self.model.predict([states_img, states_box], verbose=0)
        next_q_values = self.target_model.predict([next_states_img, next_states_box], verbose=0)

        for i in range(batch_size):
            target = rewards[i]
            if not dones[i]:
                target = (rewards[i] + self.gamma * np.amax(next_q_values[i]))
            targets[i][actions[i]] = target

        self.model.fit([states_img, states_box], targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def load_coco_dataset(folder_path):
    ann_file = os.path.join(folder_path, "_annotations.coco.json")
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    images_list = []
    boxes_list = []
    img_map = {img['id']: img['file_name'] for img in coco_data['images']}
    processed_images = set()
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id in processed_images:
            continue
        file_name = img_map.get(img_id)
        if not file_name:
            continue
        img_path = os.path.join(folder_path, file_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))
        bbox = ann['bbox'] 
        orig_w, orig_h = 0, 0
        for img_info in coco_data['images']:
            if img_info['id'] == img_id:
                orig_w = img_info['width']
                orig_h = img_info['height']
                break
        norm_bbox = [bbox[0] / orig_w, bbox[1] / orig_h, bbox[2] / orig_w, bbox[3] / orig_h]
        images_list.append(image)
        boxes_list.append(norm_bbox)
        processed_images.add(img_id)
    return np.array(images_list, dtype=np.uint8), np.array(boxes_list, dtype=np.float32)

def visualize_progress(env, agent, episode):
    num_samples = 3
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        idx = random.randint(0, len(env.images) - 1)
        img = env.images[idx]
        target_box = env.target_boxes[idx]
        env.current_img_idx = idx
        current_box = np.random.rand(4).astype(np.float32)
        state = {"image": img, "box": current_box}
        for _ in range(50):
            action = agent.act(state, training=False)
            move_size = 0.01
            if action == 0: current_box[1] -= move_size 
            elif action == 1: current_box[1] += move_size 
            elif action == 2: current_box[0] -= move_size 
            elif action == 3: current_box[0] += move_size 
            elif action == 4:
                current_box[0] -= move_size / 2
                current_box[2] += move_size
            elif action == 5:
                current_box[0] += move_size / 2
                current_box[2] -= move_size
            elif action == 6:
                current_box[1] -= move_size / 2
                current_box[3] += move_size
            elif action == 7:
                current_box[1] += move_size / 2
                current_box[3] -= move_size
            current_box = np.clip(current_box, 0, 1)
            state = {"image": img, "box": current_box}
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        rect_real = patches.Rectangle((target_box[0]*640, target_box[1]*640), target_box[2]*640, target_box[3]*640, linewidth=2, edgecolor='r', facecolor='none', label='Reale')
        rect_agent = patches.Rectangle((current_box[0]*640, current_box[1]*640), current_box[2]*640, current_box[3]*640, linewidth=2, edgecolor='g', facecolor='none', label='Agente')
        plt.gca().add_patch(rect_real)
        plt.gca().add_patch(rect_agent)
        plt.title(f"Episodio {episode}")
        plt.legend()
        plt.axis('off')
    plt.tight_layout()
    plt.show()

dataset_path = kagglehub.dataset_download("pkdarabi/brain-tumor-image-dataset-semantic-segmentation")
train_images, train_boxes = load_coco_dataset(os.path.join(dataset_path, "train"))
test_images, test_boxes = load_coco_dataset(os.path.join(dataset_path, "test"))

env_train = BrainTumorEnv(train_images, train_boxes)
env_test = BrainTumorEnv(test_images, test_boxes)
agent = DQNAgent(action_size=8)

episodes = 2000
batch_size = 64
rewards_history = [] # Lista per memorizzare la reward di ogni episodio

for e in range(episodes):
    state, _ = env_train.reset() 
    total_reward = 0
    for time in range(env_train.max_steps):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env_train.step(action)
        done = terminated or truncated
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break
    
    rewards_history.append(total_reward) # Salvataggio reward
    agent.replay(batch_size)
    
    if (e + 1) % 10 == 0:
        # Aggiorno la target network
        agent.update_target_model()

        # Calcolo la media degli ultimi 10 episodi
        avg_reward = np.mean(rewards_history[-10:])
        print(f"Episodio: {e+1}/{episodes}, Reward: {total_reward:.2f}, Media 10 ep: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    # if (e + 1) % 200 == 0:
        # visualize_progress(env_train, agent, e + 1)

print("\nInizio Fase di Test...")
test_episodes = 50
successes = 0
for e in range(test_episodes):
    state, _ = env_test.reset()
    done = False
    while not done:
        action = agent.act(state, training=False)
        next_state, reward, terminated, truncated, _ = env_test.step(action)
        state = next_state
        done = terminated or truncated
        if terminated:
            successes += 1
            break

print(f"Test completato. Success Rate: {(successes/test_episodes)*100:.2f}%")

# Implementazione grafico della reward
plt.figure(figsize=(10, 5))
plt.plot(rewards_history, label='Reward per Episodio', color='blue', alpha=0.3)
# Aggiungo una media mobile per rendere il grafico più leggibile
if len(rewards_history) >= 10:
    moving_avg = np.convolve(rewards_history, np.ones(10)/10, mode='valid')
    plt.plot(range(9, len(rewards_history)), moving_avg, label='Media Mobile (10 ep)', color='red', linewidth=2)

plt.title("Andamento della Reward durante l'addestramento")
plt.xlabel("Episodio")
plt.ylabel("Reward Totale")
plt.legend()
plt.grid(True)
plt.show()