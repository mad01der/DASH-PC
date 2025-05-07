import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.spatial import KDTree
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from collections import deque
from torch.distributions import Categorical
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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


def preprocess_data(df):
    df[['vx', 'vy', 'vz']] = df[['x', 'y', 'z']].diff().div(0.033, axis=0)
    df[['ax', 'ay', 'az']] = df[['vx', 'vy', 'vz']].diff().div(0.033, axis=0)
    df = df.fillna(0)
    return df

def load_vr_data(file_path):
    df = pd.read_csv(file_path)
    df = df.rename(columns={
        'HMDPX': 'x', 'HMDPY': 'y', 'HMDPZ': 'z',
        'HMDRX': 'rx', 'HMDRY': 'ry', 'HMDRZ': 'rz'
    }).assign(
        elapsed=lambda x: x.index * 0.033 
    )
    return preprocess_data(df)

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

class ViewportPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.to(device)
        
    def forward(self, x):
        return self.net(x)
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

  
def train_model(X_train, y_train):
    model = ViewportPredictor(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(100):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1,1))
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}') 
    return model

def evaluate_model(model, X_test, y_test):
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        preds = model(X_test_tensor).cpu().numpy().flatten()
    mse = np.mean((preds - y_test)**2)
    print(f"MSE: {mse:.4f}")
    print(f"预测准确率范围：{preds.min():.2%} - {preds.max():.2%}")
    return preds

def generate_dataset(df, window_sizes, pred_steps_list):
    samples = []
    labels = []
    for window_size, pred_steps in zip(window_sizes, pred_steps_list):
        predictor = RollingPredictor(window_size)
        detector = TurningPointDetector(window_size, pred_steps)
        for i in range(len(df)-window_size-pred_steps):
            history = df.iloc[i:i+window_size]
            latest_data = history.iloc[-1].to_dict()
            
            # 初始化特征向量
            features = []
            
            # 检查是否检测到转向点
            turning_point_index = detector.update(latest_data, 0)
            if turning_point_index is not None and turning_point_index >= 0:
                # 如果检测到转向点，从转向点开始取数据
                new_window_start = i + turning_point_index
                history = df.iloc[new_window_start:i+window_size]
            
            # 统一构建特征向量
            for _, row in history.iterrows():
                features.extend([
                    row['x'], row['y'], row['z'],  # 位置
                    row['vx'], row['vy'], row['vz'],  # 速度
                    row['ax'], row['ay'], row['az']  # 加速度
                ])
            
            # 添加窗口参数
            features.extend([window_size, pred_steps])
            
            # 确保特征向量长度一致
            expected_length = window_size * 9 + 2  # 每个时间步9个特征 + 2个参数
            if len(features) != expected_length:
                # 如果长度不一致，填充或截断
                features = features[:expected_length] if len(features) > expected_length \
                    else features + [0] * (expected_length - len(features))
            
            # 预测和评估
            predictor.window = [row.to_dict() for _, row in history.iterrows()]
            predictions = predictor.predict(pred_steps)
            actuals = df.iloc[i+window_size:i+window_size+pred_steps].to_dict('records')
            
            accuracies = []
            for pred, actual in zip(predictions, actuals):
                pred_vis = calculate_visibility(transform_tiles(pred, OBJECT_TILES))
                actual_vis = calculate_visibility(transform_tiles(actual, OBJECT_TILES))
                common = pred_vis & actual_vis
                accuracies.append(len(common)/len(actual_vis) if actual_vis else 1.0)
            
            samples.append(features)
            labels.append(accuracies[-1])
            detector.update(latest_data, np.mean(accuracies))
    
    return np.array(samples), np.array(labels)

if __name__ == '__main__':
    df = load_vr_data('../../dataset/6DoF-HMD-UserNavigationData/NavigationData/H3_nav.csv')
    df = df[df['Participant'] == 'P01_V1']
    configs = [
        (5,10)
    ]
    window_sizes, pred_steps_list = zip(*configs)
    X, y = generate_dataset(df, window_sizes, pred_steps_list)
    split_point = int(len(X) * 0.5)
    X_train = X[:split_point]  # 前50%作为测试
    y_train = y[:split_point]
    X_test = X[split_point:]  # 后50%作为训练
    y_test = y[split_point:]
    model = train_model(X_train, y_train)
    test_preds = evaluate_model(model, X_test, y_test)
    chunk_size = 5
    num_chunks = len(test_preds) // chunk_size

    for i in range(num_chunks):
    # 获取当前chunk的预测值
        chunk_start = i * chunk_size
        chunk_end = (i + 1) * chunk_size
        chunk_values = test_preds[chunk_start:chunk_end]
    
    # 计算平均值
        chunk_avg =  min(chunk_values)
    
    # 打印结果
        print(f"chunk {i+1} : value : {chunk_avg}")
    
# 处理剩余不足5个的数据（如果有）
    if len(test_preds) % chunk_size != 0:
        remaining_values = test_preds[num_chunks * chunk_size:]
        remaining_avg = sum(remaining_values) / len(remaining_values)
        print(f"chunk {num_chunks+1} : value : {remaining_avg}")
    # test_preds = evaluate_model(model, X_test, y_test)
    # plt.figure(figsize=(12,6))
    # plt.plot(y_test, label='True Value')
    # plt.plot(test_preds, label='Predict Value')
    # plt.title('VHR accuracy')
    # plt.xlabel('Label of test data')
    # plt.ylabel('VHR')
    # plt.legend()
    # plt.show() 