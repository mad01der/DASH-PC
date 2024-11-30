import * as THREE from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { OrbitControls } from "three/addons/controls/OrbitControls.js";//orbit control
import * as dat from "dat.gui";
import camera from "./src/camera";
import scene from "./src/scene";





camera.position.z = 70;
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth / 2 , window.innerHeight / 2);
renderer.domElement.style.marginTop = '170px';
document.body.appendChild(renderer.domElement);



const loader = new PLYLoader();
// try {
//   let s = '/test.ply'
//   loader.load(s,
//     function (geometry) {
//       geometry.computeVertexNormals();
//       const material = new THREE.PointsMaterial();
//       const mesh = new THREE.Points(geometry, material);
//       mesh.scale.set(0.05, 0.05, 0.05);
//       mesh.position.set(-18, -25, -10);
//       // const gui = new dat.GUI();
//       // const guiPosition = gui.addFolder("移动");
//       // guiPosition.add(mesh.position, "x").min(-20).max(-16).step(1);
//       // guiPosition.add(mesh.position, "y").min(-30).max(-20).step(1);
//       // guiPosition.add(mesh.position, "z").min(-12).max(-8).step(1);

//       // const guiScale = gui.addFolder("缩放");
//       // guiScale.add(mesh.scale, "x").min(0.05).max(0.1).step(0.01);
//       // guiScale.add(mesh.scale, "y").min(0.05).max(0.1).step(0.01);
//       // guiScale.add(mesh.scale, "z").min(0.05).max(0.1).step(0.01);

//       // const guiRotation = gui.addFolder("旋转");
//       // guiRotation.add(mesh.rotation, "x").min(-Math.PI).max(Math.PI).step(0.01);
//       // guiRotation.add(mesh.rotation, "y").min(-Math.PI).max(Math.PI).step(0.01);
//       // guiRotation.add(mesh.rotation, "z").min(-Math.PI).max(Math.PI).step(0.01);
//       scene.add(mesh);
//       scene.background = new THREE.Color(0x52645b);
//       // console.log('loader.load OK');
//     },
//     // function (xhr) {
//     //   console.log((xhr.loaded / xhr.total) * 100 + "% loaded");
//     // },
//     // function (err) {
//     //   console.error(err);
//     // }
//   );
//   // console.log('loader ok')
// }
// catch (err) {
//   //在此处理错误
//   console.log(err)
// }
// // console.log('loader ok end')
let fileIndex = 0; // 当前文件索引

// for (let i = 1450; i <= 1749; i++) {
//   fileList.push(`/redandblack_vox10_${i}_sample.ply`);
// }
// const mpdUrl = 'http://localhost/test/pointcloud.mpd';

async function fetchMpdAndFiles(fileList,mpdUrl) {
  try {
    // Fetch the MPD file
    const response = await fetch(mpdUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch MPD file: ${response.statusText}`);
    }

    // Parse the MPD XML
    const mpdText = await response.text();
    const parser = new DOMParser();
    const xmlDoc = parser.parseFromString(mpdText, "application/xml");

    // Extract file template and range
    const segmentTemplate = xmlDoc.querySelector('SegmentTemplate');
    const startNumber = parseInt(segmentTemplate.getAttribute('startNumber'), 10);
    const mediaTemplate = segmentTemplate.getAttribute('media');
    const totalFiles = 300; // Adjust based on your MPD configuration

    // Extract PLY filenames and store them in the fileList array
    for (let i = startNumber; i < startNumber + totalFiles; i++) {
      const fileName = mediaTemplate.replace('$Number$', i);

      // Check if the file has a .ply extension
      if (fileName.endsWith('.ply')) {
        fileList.push(fileName); // Add the PLY file to the list
      }
    }
    console.log("PLY files extracted:", fileList);
  } catch (error) {
    console.error(`Error fetching MPD: ${error.message}`);
  }
}


document.getElementById('startButton').addEventListener('click', function() {
  // 先清空 fileArray 数组，确保重新加载
  const fileArray = [];
  const mpdUrl = document.getElementById("mpd-select").value;
  const interval = 20;
  fetchMpdAndFiles(fileArray,mpdUrl).then(() => {
    console.log("Final fileList:", fileArray);
    let currentMesh = null;
    
    // 定义加载和渲染文件的函数
    function loadAndRenderFile(index) {
      if (index >= fileArray.length) {
        console.log(fileArray.length);
        console.log('所有文件加载完成');
        return;
      }

      const currentFile = fileArray[index];
      loader.load(
        currentFile,
        function (geometry) {
          geometry.computeVertexNormals();
          const material = new THREE.PointsMaterial({
            vertexColors: true,
            size: 0.1
          });
          const newMesh = new THREE.Points(geometry, material);
          newMesh.scale.set(0.05, 0.05, 0.05);
          newMesh.position.set(-18, -25, -10);

          // 添加到场景，但不立即移除旧的 mesh
          scene.add(newMesh);

          // 等待新点云完全加载后，移除旧点云
          if (currentMesh) {
            scene.remove(currentMesh);
          }
          currentMesh = newMesh;

          // 定时加载下一帧
          setTimeout(() => {
            loadAndRenderFile(index + 1);
          }, interval); // 根据需要调整切换时间
        },
        function (xhr) {
          console.log(`文件 ${currentFile} 加载中: ${(xhr.loaded / xhr.total) * 100}%`);
        },
        function (err) {
          console.error(`文件 ${currentFile} 加载出错:`, err);
        }
      );
    }

    try {
      loadAndRenderFile(0); // 从第一个文件开始加载
    } catch (err) {
      console.error('加载时发生错误:', err);
    }
  });
});

const gridHelper = new THREE.GridHelper(20, 20, 0xffffff, 0xffffff)
gridHelper.material.transparent = true;
gridHelper.material.opacity = 0.5;
scene.add(gridHelper);
const axesHelper = new THREE.AxesHelper(10)
scene.add(axesHelper)
const controls = new OrbitControls(camera, renderer.domElement);
renderer.render(scene, camera);





//动画
function animation() {
  controls.update();
  // 重新渲染
  renderer.render(scene, camera);
  // 下一帧渲染回调
  requestAnimationFrame(animation);
}
animation();


window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth / 2, window.innerHeight / 2);
});
