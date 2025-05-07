import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import KDTree

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

class RollingPredictor:
    def __init__(self, window_size):
        self.window_size = window_size
        self.window = []
        self.models = {}  
        self.scalers = {} 
        self.is_trained = False
        self._initialize_models() 
        
    def _initialize_models(self):
        for coord in ['x', 'y', 'z', 'rx', 'ry', 'rz']:
            model = Sequential()
            model.add(LSTM(32, activation='relu', 
                        input_shape=(self.window_size-1, 1),
                        return_sequences=False))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            self.models[coord] = model
    
    def update(self, data):
        self.window.append(data)
        if len(self.window) > self.window_size:
            self.window = self.window[-self.window_size:]  

    def _prepare_data(self, sequence):
        if not hasattr(self, '_last_sequence') or self._last_sequence != sequence:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(np.array(sequence).reshape(-1, 1))
            X = scaled_data[:-1].reshape(1, len(scaled_data)-1, 1)
            y = scaled_data[1:].reshape(1, len(scaled_data)-1, 1)
            self._last_sequence = sequence
            self._last_scaler = scaler
            self._last_X = X
            self._last_y = y
        return self._last_X, self._last_y, self._last_scaler
    
    def train_models(self):
        if len(self.window) < 2:
            return   
        for coord in ['x', 'y', 'z', 'rx', 'ry', 'rz']:
            sequence = [d[coord] for d in self.window]
            X, y, scaler = self._prepare_data(sequence)
            self.models[coord].fit(X, y, epochs=5, verbose=0)
            self.scalers[coord] = scaler
        self.is_trained = True
    
    def predict(self, steps):
        if len(self.window) < 2:
            return [self.window[-1].copy() for _ in range(steps)]
        if not self.is_trained:
            self.train_models()
        predictions = []
        current_window = {coord: np.array([d[coord] for d in self.window]) 
                         for coord in ['x', 'y', 'z', 'rx', 'ry', 'rz']}
        for _ in range(steps):
            pred = {}
            for coord in ['x', 'y', 'z', 'rx', 'ry', 'rz']:
                scaled_data = self.scalers[coord].transform(
                    current_window[coord].reshape(-1, 1))
                X = scaled_data[:-1].reshape(1, len(scaled_data)-1, 1)
                y_pred = self.models[coord].predict(X, verbose=0)[0][0]
                y_pred = self.scalers[coord].inverse_transform([[y_pred]])[0][0]
                current_window[coord] = np.roll(current_window[coord], -1)
                current_window[coord][-1] = y_pred
                pred[coord] = y_pred
            predictions.append(pred)
        return predictions

def normalize_angle(angle):
    angle = (angle + 180) % 360 - 180
    return angle

def predict_and_evaluate(df, window_size, pred_steps): 
    results = []
    predictor = RollingPredictor(window_size)
    for i in range(len(df)-window_size-pred_steps):
        print(f"\n=== 预测窗口 {i} ===")
        current_window = df.iloc[i:i+window_size]
        predictor.window = [row.to_dict() for _, row in current_window.iterrows()]
        predictions = predictor.predict(pred_steps)
        actuals = df.iloc[i+window_size:i+window_size+pred_steps].to_dict('records')
        
        # print(f"时间范围: {current_window.iloc[0]['elapsed']:.3f}s 到 {current_window.iloc[-1]['elapsed']:.3f}s")
        # print(f"预测范围: {actuals[0]['elapsed']:.3f}s 到 {actuals[-1]['elapsed']:.3f}s")
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
            rot_differ_x = abs(normalize_angle(pred['rx']) - normalize_angle(actual['rx']))
            rot_differ_y = abs(normalize_angle(pred['ry']) - normalize_angle(actual['ry']))
            differ_tile_number = (rot_differ_x / 9 * N_SPLITS + rot_differ_y / 16 * N_SPLITS - min(rot_differ_x / 9,rot_differ_y / 16) * min(rot_differ_x / 9,rot_differ_y / 16))
            common = pred_vis & actual_vis 
            true_common = max(len(common) - differ_tile_number,0)
            accuracy = true_common /len(actual_vis) if actual_vis else 1.0
            accuracies.append(accuracy)
            window_details.append({
                'step': step,
                'actual_pos': [actual['x'], actual['y'], actual['z']],
                'pred_pos': [pred['x'], pred['y'], pred['z']],
                'actual_rot': [actual['rx'], actual['ry'], actual['rz']],
                'pred_rot': [pred['rx'], pred['ry'], pred['rz']],
                'accuracy': accuracy,
                'visible_tiles': len(actual_vis),
                'predicted_tiles': len(pred_vis),
                'common_tiles': true_common
            })
        # print("\n预测步数 | 实际可见分块 | 预测可见分块 | 共同可见分块 | 准确率")
        # print("-"*80)
        # for detail in window_details:
        #     print(f"{detail['step']:2d} | {detail['visible_tiles']:4d} | "
        #           f"{detail['predicted_tiles']:4d} | {np.round(detail['common_tiles'],0)} | "
        #           f"{detail['accuracy']:.2%}")
        avg_acc = np.mean(accuracies)
        max_acc = np.max(accuracies)
        min_acc = np.min(accuracies)
        last_step_detail = window_details[-1]
        # print(f"\n窗口统计: 平均准确率={avg_acc:.2%}, 最高准确率={max_acc:.2%}, 最后准确率={last_step_detail['accuracy']:.2%}")
        results.append({
            'start_time': df.iloc[i]['elapsed'],
            'avg_accuracy': avg_acc,
            'max_accuracy': max_acc,
            'min_accuracy': min_acc,
            'last_step_accuracy': last_step_detail['accuracy']
        }) 
    return pd.DataFrame(results)

def visualize_results(results):
    plt.figure(figsize=(14, 6))
    # plt.plot(results['start_time'], results['avg_accuracy'], 'b-', label='Average Accuracy', alpha=0.7)
    # plt.plot(results['start_time'], results['max_accuracy'], 'g--', label='Max Accuracy', alpha=0.5)
    plt.plot(results['start_time'], results['last_step_accuracy'], 'r--', label='Accuracy', alpha=0.5)
    plt.title('Prediction Accuracy Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    df = load_vr_data('../../dataset/6DoF-HMD-UserNavigationData/NavigationData/H3_nav.csv') # 4,8,12,16,20,32,40,60

    results_1 = predict_and_evaluate(df[df['Participant'] == 'P01_V1'],4,4)
    print("result 1 is:",np.mean(results_1['last_step_accuracy']))

    results_2 = predict_and_evaluate(df[df['Participant'] == 'P01_V1'],8,8)
    print("result 2 is:",np.mean(results_2['last_step_accuracy']))

    results_3 = predict_and_evaluate(df[df['Participant'] == 'P01_V1'],12,12)
    print("result 3 is:",np.mean(results_3['last_step_accuracy']))
    
    results_4 = predict_and_evaluate(df[df['Participant'] == 'P01_V1'],16,16)
    print("result 4 is:",np.mean(results_4['last_step_accuracy']))

    results_5 = predict_and_evaluate(df[df['Participant'] == 'P01_V1'],20,20)
    print("result 5 is:",np.mean(results_5['last_step_accuracy']))

    results_6 = predict_and_evaluate(df[df['Participant'] == 'P01_V1'],32,32)
    print("result 6 is:",np.mean(results_6['last_step_accuracy']))

    results_7 = predict_and_evaluate(df[df['Participant'] == 'P01_V1'],40,40)
    print("result 7 is:",np.mean(results_7['last_step_accuracy']))

    results_8 = predict_and_evaluate(df[df['Participant'] == 'P01_V1'],60,60)
    print("result 8 is:",np.mean(results_8['last_step_accuracy']))