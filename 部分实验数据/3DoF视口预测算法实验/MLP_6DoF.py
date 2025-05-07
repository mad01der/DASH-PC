import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.spatial import KDTree
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

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
    # df[['vrx', 'vry', 'vrz']] = df[['rx', 'ry', 'rz']].diff().div(0.033, axis=0)
    # df[['arx', 'ary', 'arz']] = df[['vrx', 'vry', 'vrz']].diff().div(0.033, axis=0)
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
            return [self.window[-1]] * steps if self.window else []
        
        X = np.arange(len(self.window)).reshape(-1, 1)  
        predictions = []
        for i in range(steps):
            pred = {}
            for coord in ['x', 'y', 'z', 'rx', 'ry', 'rz']:
                y = np.array([d[coord] for d in self.window]).reshape(-1)
                model = LinearRegression()
                model.fit(X, y)
                next_time = np.array([[len(self.window) + i]]) 
                pred[coord] = model.predict(next_time)[0] 
            predictions.append(pred)
        return predictions

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

def normalize_angle(angle):
    angle = (angle + 180) % 360 - 180
    return angle

def generate_dataset(df, window_sizes, pred_steps_list):
    samples = []
    labels = []
    for window_size, pred_steps in zip(window_sizes, pred_steps_list):
        predictor = RollingPredictor(window_size)
        for i in range(len(df)-window_size-pred_steps):
            history = df.iloc[i:i+window_size]
            features = []
            for _, row in history.iterrows():
                features.extend([
                    row['x'], row['y'], row['z'],
                    row['vx'], row['vy'], row['vz'],
                    row['ax'], row['ay'], row['az'],
                    # row['rx'], row['ry'], row['rz'],
                    # row['vrx'], row['vry'], row['vrz'],
                    # row['arx'], row['ary'], row['arz'],
                ])
            features.extend([window_size, pred_steps])
            predictor.window = [row.to_dict() for _, row in history.iterrows()]
            predictions = predictor.predict(pred_steps)
            actuals = df.iloc[i+window_size:i+window_size+pred_steps].to_dict('records')
            accuracies = []
            for pred, actual in zip(predictions, actuals):
                pred_vis = calculate_visibility(transform_tiles(pred, OBJECT_TILES))
                actual_vis = calculate_visibility(transform_tiles(actual, OBJECT_TILES))
                rot_differ_x = abs(normalize_angle(pred['rx']) - normalize_angle(actual['rx']))
                rot_differ_y = abs(normalize_angle(pred['ry']) - normalize_angle(actual['ry']))
                differ_tile_number = (rot_differ_x / 9 * N_SPLITS + rot_differ_y / 16 * N_SPLITS - 
                                 min(rot_differ_x / 9, rot_differ_y / 16) * min(rot_differ_x / 9, rot_differ_y / 16))
                common = pred_vis & actual_vis
                true_common = max(len(common) - differ_tile_number, 0)
                accuracy = true_common / len(actual_vis) if actual_vis else 1.0
                accuracies.append(accuracy)
            samples.append(features)
            labels.append(accuracies[-1])
    return np.array(samples), np.array(labels)

if __name__ == '__main__':
    df = load_vr_data('../../dataset/6DoF-HMD-UserNavigationData/NavigationData/H3_nav.csv')
    df = df[df['Participant'] == 'P01_V1']
    configs = [
        (8,16)
    ]
    window_sizes, pred_steps_list = zip(*configs)
    X, y = generate_dataset(df, window_sizes, pred_steps_list)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)
    model = train_model(X_train, y_train)
    test_preds = evaluate_model(model, X_test, y_test)
    plt.figure(figsize=(12,6))
    plt.plot(y_test, label='True Value')
    plt.plot(test_preds, label='Predict Value')
    plt.title('VHR accuracy')
    plt.xlabel('Label of test data')
    plt.ylabel('VHR')
    plt.legend()
    plt.show()