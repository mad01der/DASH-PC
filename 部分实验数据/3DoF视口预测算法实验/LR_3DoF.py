import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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

def predict_and_evaluate(df, window_size=16, pred_steps=8): 
    results = []
    predictor = RollingPredictor(window_size)
    for i in range(len(df)-window_size-pred_steps):
        current_window = df.iloc[i:i+window_size]
        predictor.window = [row.to_dict() for _, row in current_window.iterrows()]
        predictions = predictor.predict(pred_steps)
        actuals = df.iloc[i+window_size:i+window_size+pred_steps].to_dict('records')
        # print(f"\n=== 预测窗口 {i} ===")
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
            common = pred_vis & actual_vis 
            true_common = max(len(common),0)
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
        # print(f"\n窗口统计: 平均准确率={avg_acc:.2%}, 最高准确率={max_acc:.2%}, 最低准确率={min_acc:.2%}")
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
    df = load_vr_data('../../dataset/6DoF-HMD-UserNavigationData/NavigationData/H3_nav.csv')
    results_1 = predict_and_evaluate(df[df['Participant'] == 'P01_V1'],6,12)
    print("result 1 is:",np.mean(results_1['last_step_accuracy']))

    # results_2 = predict_and_evaluate(df[df['Participant'] == 'P01_V1'],4,8)
    # print("result 2 is:",np.mean(results_2['last_step_accuracy']))

    # results_3 = predict_and_evaluate(df[df['Participant'] == 'P01_V1'],6,12)
    # print("result 3 is:",np.mean(results_3['last_step_accuracy']))
    
    # results_4 = predict_and_evaluate(df[df['Participant'] == 'P01_V1'],8,16)
    # print("result 4 is:",np.mean(results_4['last_step_accuracy']))

    # results_5 = predict_and_evaluate(df[df['Participant'] == 'P01_V1'],10,20)
    # print("result 5 is:",np.mean(results_5['last_step_accuracy']))

    # results_6 = predict_and_evaluate(df[df['Participant'] == 'P01_V1'],16,32)
    # print("result 6 is:",np.mean(results_6['last_step_accuracy']))

    # results_7 = predict_and_evaluate(df[df['Participant'] == 'P01_V1'],20,40)
    # print("result 7 is:",np.mean(results_7['last_step_accuracy']))

    # results_8 = predict_and_evaluate(df[df['Participant'] == 'P01_V1'],30,60)
    # print("result 8 is:",np.mean(results_8['last_step_accuracy']))
