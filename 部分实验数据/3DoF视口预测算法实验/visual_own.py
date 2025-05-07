import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = "../location_record/record_1.txt"

pattern = re.compile(
    r"position: \{ x: (?P<x>-?\d+\.\d+), y: (?P<y>-?\d+\.\d+), z: (?P<z>-?\d+\.\d+) \},"
    r"\s+rotation: \{ x: (?P<rx>-?\d+\.\d+), y: (?P<ry>-?\d+\.\d+), z: (?P<rz>-?\d+\.\d+) \}\s+\}"
)
data = []
with open(file_path, "r") as file:
    for match in pattern.finditer(file.read()):
        data.append({
            "position": {
                "x": float(match.group("x")),
                "y": float(match.group("y")),
                "z": float(match.group("z"))
            },
            "rotation": {
                "x": float(match.group("rx")),
                "y": float(match.group("ry")),
                "z": float(match.group("rz"))
            }
        })
df = pd.DataFrame([{
    'x': d['position']['x'],
    'y': d['position']['y'],  
    'z': d['position']['z'],
    'rx': d['rotation']['z'],
    'ry': d['rotation']['y'],
    'rz': d['rotation']['x']
} for d in data])
def calculate_direction(rx, ry, rz):
    direction = np.array([0, 0, -1])
    rotation_matrix = np.array([
        [np.cos(ry) * np.cos(rz), -np.cos(ry) * np.sin(rz), np.sin(ry)],
        [np.sin(rx) * np.sin(ry) * np.cos(rz) + np.cos(rx) * np.sin(rz),
         -np.sin(rx) * np.sin(ry) * np.sin(rz) + np.cos(rx) * np.cos(rz),
         -np.sin(rx) * np.cos(ry)],
        [-np.cos(rx) * np.sin(ry) * np.cos(rz) + np.sin(rx) * np.sin(rz),
         np.cos(rx) * np.sin(ry) * np.sin(rz) + np.sin(rx) * np.cos(rz),
         np.cos(rx) * np.cos(ry)]
    ])
    direction = np.dot(rotation_matrix, direction)
    return direction
# df['direction'] = df.apply(lambda row: calculate_direction(row['rx'], row['ry'], row['rz']), axis=1)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(df['x'], df['y'], df['z'], color='green', linewidth=1)
ax.scatter(df['x'], df['y'], df['z'], color='blue', s=3, label='Position Points')
# for i in range(0, len(df), 1):
#     pos = df.iloc[i]
#     direction = pos['direction']
#     ax.quiver(
#         pos['x'], pos['y'], pos['z'],
#         direction[0], direction[1], direction[2],
#         length=0.1, color='red', arrow_length_ratio=0.2,
#         linewidth=1
#     )
ax.scatter([0], [1.4], [-1], color='red', s=100, marker='D', label='Object Center')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('PICO trajectory')
ax.legend()
plt.show()