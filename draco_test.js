import * as THREE from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { OrbitControls } from "three/addons/controls/OrbitControls.js"; // orbit control
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import camera from "./src/camera";
import scene from "./src/scene";

camera.position.z = 70;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth / 2 , window.innerHeight / 2);
renderer.domElement.style.marginTop = '170px';
document.body.appendChild(renderer.domElement);


const loader = new DRACOLoader();
loader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.6/');
loader.setDecoderConfig({ type: 'js' });
loader.preload();
// const loader = new PLYLoader();
// 文件列表
const fileArray = [];
for (let i = 1450; i <= 1749; i++) {
  fileArray.push(`/drc/redandblack_vox10_${i}.drc`);
}
let currentIndex = 0; // 当前加载文件的索引
let currentMesh = null;

// 加载并渲染点云文件
function loadAndRenderFile(index) {
  if (index >= fileArray.length) {
    console.log('所有文件加载完成');
    return;
  }

  const currentFile = fileArray[index];
  loader.load(
    currentFile,
    function (geometry) {
      geometry.computeVertexNormals();

      // 获取原始位置数据

      // 创建材质和网格
      const material = new THREE.PointsMaterial({
        vertexColors: true,
        size: 0.01
      });

      const newMesh = new THREE.Points(geometry, material);
      newMesh.scale.set(0.05, 0.05, 0.05);
      newMesh.position.set(-18, -25, -10);

      // 打印当前加载的模型的点数
      const pointCount = geometry.attributes.position.count;
      console.log(`文件 ${currentFile} 加载完成，点数: ${pointCount}`);

      // 等待新点云完全加载后，移除旧的 mesh
      if (currentMesh) {
        scene.remove(currentMesh);
      }
      scene.add(newMesh);
      currentMesh = newMesh;

      // 定时加载下一帧
      setTimeout(() => {
        loadAndRenderFile(index + 1);
      }, 20); // 根据需要调整切换时间
    },
    function (xhr) {
      console.log(`文件 ${currentFile} 加载中: ${(xhr.loaded / xhr.total) * 100}%`);
    },
    function (err) {
      console.error(`文件 ${currentFile} 加载出错:`, err);
    }
  );
}

// 添加辅助工具（网格和坐标轴）
const gridHelper = new THREE.GridHelper(20, 20, 0xffffff, 0xffffff);
gridHelper.material.transparent = true;
gridHelper.material.opacity = 0.5;
scene.add(gridHelper);

const axesHelper = new THREE.AxesHelper(10);
scene.add(axesHelper);

const controls = new OrbitControls(camera, renderer.domElement);

// 初始化渲染循环
function animation() {
  controls.update(); // 更新 OrbitControls
  renderer.render(scene, camera); // 渲染场景
  requestAnimationFrame(animation); // 下一帧渲染回调
}

animation(); // 启动动画循环

// 在窗口大小变化时更新渲染器和相机
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth / 2, window.innerHeight / 2);
});

// 开始加载第一个文件
loadAndRenderFile(0);