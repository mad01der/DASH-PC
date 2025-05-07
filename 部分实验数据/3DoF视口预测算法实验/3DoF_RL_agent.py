import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.spatial import KDTree

from scipy.stats import gaussian_kde
import matplotlib.style as mplstyle
OBJECT_VERTICES = np.array([
    [-0.3, 1.1, -0.3], [0.3, 1.1, -0.3],
    [-0.3, 2.3, -0.3], [0.3, 2.3, -0.3],
    [-0.3, 1.1, 0.3], [0.3, 1.1, 0.3],
    [-0.3, 2.3, 0.3], [0.3, 2.3, 0.3]
])
N_SPLITS = 5
VISIBLE = 0.988
OBJECT_TILES = []
min_coords = OBJECT_VERTICES.min(axis=0)
max_coords = OBJECT_VERTICES.max(axis=0)
steps = (max_coords - min_coords) / N_SPLITS

for i in range(N_SPLITS):
    for j in range(N_SPLITS):
        for k in range(N_SPLITS):
            center = min_coords + steps * [i+0.5, j+0.5, k+0.5]
            radius = np.linalg.norm(steps/2) 
            OBJECT_TILES.append({
                'index': len(OBJECT_TILES),
                'center': center,
                'radius': radius
            })
TREE = KDTree([t['center'] for t in OBJECT_TILES])

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states), np.array(self.actions), np.array(self.probs), \
               np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions, alpha):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(n_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.to(self.device)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        dist = Categorical(logits=self.fc3(x))
        return dist
    
    def save_checkpoint(self, filename):
        torch.save(self.state_dict(), filename)
        
    def load_checkpoint(self, filename):
        self.load_state_dict(torch.load(filename))

class CriticNetwork(nn.Module):
    def __init__(self, n_states, alpha):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(n_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value
    
    def save_checkpoint(self, filename):
        torch.save(self.state_dict(), filename)
        
    def load_checkpoint(self, filename):
        self.load_state_dict(torch.load(filename))

class PPOAgent:
    def __init__(self, n_states, n_actions, gamma=0.99, alpha=0.0003, 
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        
        self.actor = ActorNetwork(n_states, n_actions, alpha)
        self.critic = CriticNetwork(n_states, alpha)
        self.memory = PPOMemory(batch_size)
        
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)
    
    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float).to(self.actor.device) 
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        return action, probs, value
    
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
                self.memory.generate_batches()
            
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k+1] * \
                          (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
                
            advantage = torch.tensor(advantage).to(self.actor.device)
            values = torch.tensor(values).to(self.actor.device)
            
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                                                   1+self.policy_clip) * advantage[batch]
                
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value).pow(2).mean()
                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                
        self.memory.clear_memory()

class RollingPredictor:
    def __init__(self, window_size):
        self.window = []
        self.window_size = window_size
    def update(self, data):
        self.window.append(data)
        if len(self.window) > self.window_size:
            self.window.pop(0)
    def predict(self, steps):
        if len(self.window) < 2:
            return [self.window[-1]] * steps
        X = np.arange(len(self.window)).reshape(-1, 1)
        models = {}
        for coord in ['x', 'y', 'z', 'rx', 'ry', 'rz']:
            model = LinearRegression()
            model.fit(X, [d[coord] for d in self.window])
            models[coord] = model
        return [{
            'x': models['x'].predict([[len(self.window)+i]])[0],
            'y': models['y'].predict([[len(self.window)+i]])[0],
            'z': models['z'].predict([[len(self.window)+i]])[0],
            'rx': models['rx'].predict([[len(self.window)+i]])[0],
            'ry': models['ry'].predict([[len(self.window)+i]])[0],
            'rz': models['rz'].predict([[len(self.window)+i]])[0],
        } for i in range(steps)]

class TurningPointDetector:
    def __init__(self, window_size, pred_steps):
        self.window_size = window_size
        self.pred_steps = pred_steps
        self.history = deque(maxlen=window_size)
        self.agent = PPOAgent(n_states=9, n_actions=2)  
        
    def extract_features(self, new_data_point):
        if not self.history:
            return np.zeros(9) 
        current = new_data_point
        prev = self.history[-1]
        pos_features = np.array([
        current['x'], current['y'], current['z'],
        current['x'] - prev['x'], 
        current['y'] - prev['y'], 
        current['z'] - prev['z']
        ])
        if len(self.history) > 1:
            prev_prev = self.history[-2]
            accel_features = np.array([
            (current['x'] - prev['x']) - (prev['x'] - prev_prev['x']),  
            (current['y'] - prev['y']) - (prev['y'] - prev_prev['y']),  
            (current['z'] - prev['z']) - (prev['z'] - prev_prev['z'])   
            ])
        else:
            accel_features = np.zeros(3) 
        features = np.concatenate([pos_features, accel_features])
        return features
    
    def calculate_reward(self, action, prediction_accuracy):
        if action == 1:
            if len(self.history) > 1:
                prev_acc = self.history[-2].get('accuracy', 0.5)
                if(prediction_accuracy - prev_acc > 0):
                   reward = (prediction_accuracy - prev_acc) * 2
                else:
                   reward = (prediction_accuracy - prev_acc) 
            else:
                reward = prediction_accuracy - 0.5
        else:
            if len(self.history) > 1:
                prev_acc = self.history[-2].get('accuracy', 0.5)
                reward = 0.5 * (prediction_accuracy - prev_acc)
            else:
                reward = 0
        return reward
    
    def update(self, new_data_point, prediction_accuracy):
        state = self.extract_features(new_data_point)
        action, prob, val = self.agent.choose_action(state)
        reward = self.calculate_reward(action, prediction_accuracy)
        done = False  
        self.agent.remember(state, action, prob, val, reward, done)
        new_data_point['accuracy'] = prediction_accuracy
        self.history.append(new_data_point)
        if len(self.history) % self.window_size == 0:
            self.agent.learn()
        if action == 1: 
            if len(self.history) > 1:
                changes = []
                for i in range(1, len(self.history)):
                    prev = self.history[i-1]
                    curr = self.history[i]
                    change = np.sqrt(
                        (curr['x']-prev['x'])**2 + 
                        (curr['y']-prev['y'])**2 + 
                        (curr['z']-prev['z'])**2
                    )
                    changes.append(change)
                if changes:
                    max_change_idx = np.argmax(changes)
                    return max_change_idx
            return 0 
        return None 

def load_vr_data(file_path):
    df = pd.read_csv(file_path)
    return df.rename(columns={
        'HMDPX': 'x', 'HMDPY': 'y', 'HMDPZ': 'z',
        'HMDRX': 'rx', 'HMDRY': 'ry', 'HMDRZ': 'rz'
    }).assign(
        elapsed=lambda x: x.index * 0.033 
    )

def transform_tiles(position, tiles):
    offset = np.array([position['x'], position['y'], position['z']])
    return [{
        'index': t['index'],
        'center':(t['center'] - offset),
        'radius': t['radius']
    } for t in tiles]

def calculate_visibility(transformed_tiles):
    centers = np.array([t['center'] for t in transformed_tiles])
    distances = np.linalg.norm(centers, axis=1)
    sorted_indices = np.argsort(distances)
    visible = set()
    occupied_indices = set() 
    for idx in sorted_indices:
        tile = transformed_tiles[idx]
        is_visible = True
        for occupied_idx in occupied_indices:
            other = transformed_tiles[occupied_idx]
            dot_product = np.dot(tile['center'], other['center'])
            norm_product = np.linalg.norm(tile['center']) * np.linalg.norm(other['center'])
            if dot_product > VISIBLE * norm_product:
                is_visible = False
                break      
        if is_visible:
            visible.add(tile['index'])
            occupied_indices.add(idx)  
    return visible

def predict_with_turning_points(df, window_size, pred_steps):
    detector = TurningPointDetector(window_size, pred_steps)
    predictor = RollingPredictor(window_size)
    results = []
    accuracy_values = []
    for i in range(len(df)-window_size-pred_steps):
        current_window = df.iloc[i:i+window_size]
        latest_data = current_window.iloc[-1].to_dict()
        turning_point_index = detector.update(latest_data, 0)
        if turning_point_index is not None and turning_point_index >= 0:
            new_window_start = i + turning_point_index
            current_window = df.iloc[new_window_start:i+window_size]  
            predictor.window = [row.to_dict() for _, row in current_window.iterrows()]
        else:
            predictor.window = [row.to_dict() for _, row in current_window.iterrows()]  
        predictions = predictor.predict(pred_steps)
        actuals = df.iloc[i+window_size:i+window_size+pred_steps].to_dict('records')
        accuracies = []
        window_details = []
        for step, (pred, actual) in enumerate(zip(predictions, actuals), 1):
            pred_tiles = transform_tiles(
                position={'x': pred['x'], 'y': pred['y'], 'z': pred['z']},
                tiles=OBJECT_TILES
            )
            actual_tiles = transform_tiles(     
                position={'x': actual['x'], 'y': actual['y'], 'z': actual['z']},
                tiles=OBJECT_TILES
            )
            pred_vis = calculate_visibility(pred_tiles)
            actual_vis = calculate_visibility(actual_tiles)
            common = pred_vis & actual_vis 
            true_common = max(len(common),0)
            accuracy = true_common /len(actual_vis) if actual_vis else 1.0
            accuracies.append(accuracy)
            window_details.append({
                'step': step,
                'actual_pos': [actual['x'], actual['y'], actual['z']],
                'pred_pos': [pred['x'], pred['y'], pred['z']],
                'accuracy': accuracy,
                'visible_tiles': len(actual_vis),
                'predicted_tiles': len(pred_vis),
                'common_tiles': true_common
            })
            
        avg_acc = np.mean(accuracies)
        max_acc = np.max(accuracies)
        min_acc = np.min(accuracies)
        last_step_detail = window_details[-1]
        accuracy_values.append(last_step_detail['accuracy'])
        detector.update(latest_data, avg_acc)
        results.append({
            'start_time': df.iloc[i]['elapsed'],
            'avg_accuracy': avg_acc,
            'max_accuracy': max_acc,
            'min_accuracy': min_acc,
            'last_step_accuracy': last_step_detail['accuracy']
        })
        window_size_smooth = 8  # 滑动窗口大小
        accuracy_values_smoothed = np.convolve(
            accuracy_values, 
            np.ones(window_size_smooth)/window_size_smooth, 
            mode='valid'
        )
    
    return pd.DataFrame(results),accuracy_values_smoothed

def plot_cdf(data_list, labels):
    mplstyle.use(['seaborn-v0_8-poster', 'seaborn-v0_8-whitegrid'])
    plt.rcParams['axes.edgecolor'] = '#333F4B'
    plt.rcParams['axes.linewidth'] = 1.5
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_list)))
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    
    for i, (data, label) in enumerate(zip(data_list, labels)):
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
        
        kde = gaussian_kde(sorted_data)
        x_smooth = np.linspace(min(sorted_data), max(sorted_data), 200)
        y_smooth = np.interp(x_smooth, sorted_data, cdf)
        
        ax.plot(x_smooth, y_smooth, 
                color=colors[i],
                linestyle=line_styles[i % len(line_styles)],
                linewidth=2,
                label=f'{label}',
                alpha=0.9)
    
    ax.set_xlabel('VHR', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    legend = ax.legend(fontsize=10, framealpha=1, shadow=True,
                      bbox_to_anchor=(1.05, 1), loc='upper left')
    legend.get_frame().set_edgecolor('#333F4B')
    
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('#333F4B')
    
    plt.tight_layout()
    return fig

# def visualize_results(results):
#     plt.figure(figsize=(14, 6))
#     plt.plot(results['start_time'], results['last_step_accuracy'], 'r--', label='Accuracy', alpha=0.5)
#     plt.title('Prediction Accuracy Over Time (Position-only Features)')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Accuracy')
#     plt.ylim(0, 1)
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     plt.show()

if __name__ == '__main__':
    df = load_vr_data('../../dataset/6DoF-HMD-UserNavigationData/NavigationData/H3_nav.csv')
    participant_df = df[df['Participant'] == 'P01_V1']
    
    configs = [
        (6, 12), (8, 16), 
        (10, 20), (16, 32)
    ]
    labels = [f' PS={p}' for w, p in configs]
    
    accuracy_data = []
    
    for window_size, pred_steps in configs:
        _, acc_values = predict_with_turning_points(participant_df, window_size, pred_steps)
        accuracy_data.append(acc_values)
        print(f"Config WS={window_size}, PS={pred_steps}: Mean accuracy = {np.mean(acc_values):.3f}")
    
    # Plot all CDFs together
    fig = plot_cdf(accuracy_data, labels)
    fig.savefig('combined_cdf_3dof_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)