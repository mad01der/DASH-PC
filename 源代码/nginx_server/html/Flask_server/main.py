from flask import Flask, request, jsonify
from flask_cors import CORS 
from threading import Lock
import subprocess
import sys
import os
import shutil
import time
import view_prediction as vp
app = Flask(__name__)
CORS(app, origins='*')  

log_buffer = []
buffer_lock = Lock()
current_task_id = 0
logs = [] 

def clear_logs():
    global log_buffer, current_task_id
    with buffer_lock:
        log_buffer.clear()
        current_task_id = int(time.time())  

def append_log(log):
    with buffer_lock:
        log_buffer.append(log)


def copy_folder(src_folder, dest_folder):
    if os.path.exists(src_folder):
        print(f"📦 Copying {src_folder} to {dest_folder}...")
        if os.path.exists(dest_folder):
            print(f"⚠️ Target folder {dest_folder} exists. Clearing its contents...")
            try:
                shutil.rmtree(dest_folder)
                print(f"✅ Target folder {dest_folder} cleared.")
            except Exception as e:
                print(f"❌ Failed to clear folder {dest_folder}: {e}")
                return False
        try:
            shutil.copytree(src_folder, dest_folder)
            print(f"✅ Folder copied successfully!")
        except Exception as e:
            print(f"❌ Failed to copy folder: {e}")
            return False
    else:
        print(f"❌ Source folder {src_folder} does not exist.")
        return False
    return True

@app.route('/api/logs', methods=['GET'])
def get_logs():
    since = request.args.get('since', default=0, type=int)
    filtered_logs = logs[since:] 
    return jsonify({"logs": filtered_logs, "last_id": len(logs)})

data_x = [] ; data_y = [] ;data_z = [] ; data_number = []
@app.route('/position-prediction', methods=['POST'])
def position_prediction():
    #*算法2:视口预测算法服务器端接口*#
    global data_x, data_y, data_z
    try:
        data = request.get_json()
        data_x.append(data['x']);data_y.append(data['y']);data_z.append(data['z']) ;data_number.append(data['frame'])
        log_message = f"the {data['frame']  - 1450 + 1}'s camera's positions is {data['x'],data['y'],data['z']}"
        logs.append(log_message)  
        print(log_message)
        if len(data_x) == 5 and len(data_y) == 5 and len(data_z) == 5:
            vp_logs = vp.main(data_x,data_y,data_z,data_number) 
            logs.extend(vp_logs)
            data_x.clear();data_y.clear();data_z.clear();data_number.clear()
        if len(logs) == 200:
            time.sleep(2)
            logs.clear()
        return jsonify({"status": "success", "message": "预测已经完成"})
    except Exception as e:
        print("⚠️ 数据解析错误:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 400

    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)



