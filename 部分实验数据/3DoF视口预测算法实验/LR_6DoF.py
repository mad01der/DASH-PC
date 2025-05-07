import numpy as np
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
        'center': (t['center'] - offset),
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

def normalize_angle(angle):
    angle = (angle + 180) % 360 - 180
    return angle

def predict_and_evaluate(df, window_size=5, pred_steps=40):
    results = []
    accuracy_values = []
    predictor = RollingPredictor(window_size)
    
    for i in range(len(df)-window_size-pred_steps):
        current_window = df.iloc[i:i+window_size]
        predictor.window = [row.to_dict() for _, row in current_window.iterrows()]
        predictions = predictor.predict(pred_steps)
        actuals = df.iloc[i+window_size:i+window_size+pred_steps].to_dict('records')
        
        window_accuracies = []
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
            differ_tile_number = (rot_differ_x / 9 * N_SPLITS + rot_differ_y / 16 * N_SPLITS - 
                                 min(rot_differ_x / 9, rot_differ_y / 16) * min(rot_differ_x / 9, rot_differ_y / 16))
            
            common = pred_vis & actual_vis 
            true_common = max(len(common) - differ_tile_number, 0)
            accuracy = true_common / len(actual_vis) if actual_vis else 1.0
            window_accuracies.append(accuracy)
        
        
        avg_acc = np.mean(window_accuracies)
        max_acc = np.max(window_accuracies)
        min_acc = np.min(window_accuracies)
        last_step_acc = window_accuracies[-1]
        accuracy_values.append(last_step_acc)
        results.append({
            'start_time': df.iloc[i]['elapsed'],
            'avg_accuracy': avg_acc,
            'max_accuracy': max_acc,
            'min_accuracy': min_acc,
            'last_step_accuracy': last_step_acc
        })
    
    return pd.DataFrame(results), accuracy_values

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
                # label=f'{label} (μ={np.mean(data):.3f})',
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

def main():
    # change your data here
    df = load_vr_data('../../dataset/6DoF-HMD-UserNavigationData/NavigationData/H3_nav.csv')
    participant_df = df[df['Participant'] == 'P01_V1']
    
    configs = [
        (6, 12), (8, 16), 
        (10, 20), (16, 32)
    ]
    labels = [f'WS={w}, PS={p}' for w, p in configs]
    
    accuracy_data = []
    
    for window_size, pred_steps in configs:
        _, acc_values = predict_and_evaluate(participant_df, window_size, pred_steps)
        accuracy_data.append(acc_values)
        print(f"Config WS={window_size}, PS={pred_steps}: Mean accuracy = {np.mean(acc_values):.3f}")
    
    # Plot all CDFs together
    fig = plot_cdf(accuracy_data, labels)
    fig.savefig('combined_cdf_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    main()