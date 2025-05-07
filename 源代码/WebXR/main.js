import * as THREE from 'three';
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import camera from "./src/camera";
import scene from "./src/scene";
import { fetchAndParseMPD, play_control } from "./play_control";

export function init3DView(container) {
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    canvas: document.querySelector('canvas') || document.createElement('canvas')
  });
  renderer.xr.enabled = true;
  renderer.setPixelRatio(window.devicePixelRatio);
  const vrButton = VRButton.createButton(renderer);
  let dataCount = 0; 
  const MAX_DATA_COUNT = 1500; 
  vrButton.addEventListener('click', () => {
    setInterval(updateVrStatus, 20);
  });
  document.body.appendChild(vrButton);
  camera.position.set(0, 0, 65);
  if (!document.querySelector('canvas')) {
    renderer.domElement.style.margin = "4px 15px";
    renderer.setSize(window.innerWidth/1.5, window.innerHeight/1.5);
    container.appendChild(renderer.domElement);
  }
  const loader = new DRACOLoader();
  loader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.6/');
  loader.setDecoderConfig({ type: 'js' });
  loader.preload();
  document.getElementById('startButton').addEventListener('click', async function () {
    const mpdUrl = document.getElementById("mpd-input").value;
    const representations = await fetchAndParseMPD(mpdUrl);
    play_control(representations);
  });
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enabled = !renderer.xr.isPresenting;
  // let dataBuffer = [];
  // const BATCH_SIZE = 100;
  async function updateVrStatus() {
    if (dataCount >= MAX_DATA_COUNT) {
      console.log("Stopped sending data: Reached 2000 entries.");
      return; 
    }
    const pos = { x: camera.position.x, y: cameraContainer.position.y, z: camera.position.z };
    const rot = camera.rotation;
    const currentData = {
      position: {
        x: parseFloat(pos.x.toFixed(3)),
        y: parseFloat(pos.y.toFixed(3)),
        z: parseFloat(pos.z.toFixed(3))
      },
      rotation: {
        x: parseFloat(rot.x.toFixed(3)),
        y: parseFloat(rot.y.toFixed(3)),
        z: parseFloat(rot.z.toFixed(3))
      }
    };
    // dataBuffer.push(currentData);
    // dataCount++;
    // if (dataBuffer.length >= BATCH_SIZE) {
    //   await sendBatchData();
    // }
    // try {
    //   const response = await fetch('https://localhost:3000/vr-data', {
    //     method: 'POST',
    //     headers: {
    //       'Content-Type': 'application/json'
    //     },
    //     body: JSON.stringify(currentData)
    //   });
    //   if (!response.ok) {
    //     throw new Error(`HTTP error! status: ${response.status}`);
    //   }
    //   dataCount++;
    //   console.log(dataCount)
    // } catch (error) {
    //   console.error('Failed to send data:', error);
    // }
  }
  // async function sendBatchData() {
  //   if (dataBuffer.length === 0) return; // 无数据时不发送
  //   try {
  //     const response = await fetch('https://localhost:3000/vr-data', {
  //       method: 'POST',
  //       headers: {
  //         'Content-Type': 'application/json'
  //       },
  //       body: JSON.stringify(dataBuffer) // 发送整个数组
  //     });
  
  //     if (!response.ok) {
  //       throw new Error(`HTTP error! status: ${response.status}`);
  //     }
  
  //     console.log(`Successfully sent ${dataBuffer.length} entries. Total sent: ${dataCount}`);
  //     dataBuffer = []; // 清空缓存
  //   } catch (error) {
  //     console.error('Failed to send batch data:', error);
  //   }
  // }
  const controller1 = renderer.xr.getController(1); 
  controller1.addEventListener('selectstart', onSelectStart);
  controller1.addEventListener('selectend', onSelectEnd);
  controller1.addEventListener('squeezestart', onSqueezeStart)
  controller1.addEventListener('squeezeend', onSqueezeEnd)

  let isTriggerPressed_select = false;
  let isTriggerPressed_squeeze = false;
  let initialControllerY = 0;
  const cameraContainer = new THREE.Group();
  scene.add(cameraContainer);
  cameraContainer.add(camera);
  function onSelectStart(event) {
    console.log('onSelectStart', event);
    isTriggerPressed_select = true;
    const worldPos = new THREE.Vector3();
    controller1.getWorldPosition(worldPos);
    initialControllerY = cameraContainer.worldToLocal(worldPos).y;
  }
  function onSelectEnd(event) {
    console.log('onSelectEnd', event);
    isTriggerPressed_select = false;
  }
  function onSqueezeStart(event) {
    console.log('onSelectStart', event);
    isTriggerPressed_squeeze = true;
    const worldPos = new THREE.Vector3();
    controller1.getWorldPosition(worldPos);
    initialControllerY = cameraContainer.worldToLocal(worldPos).y;
  }
  function onSqueezeEnd(event) {
    console.log('onSelectEnd', event);
    isTriggerPressed_squeeze = false;
  }
  function animate() {
    if (!renderer.xr.isPresenting) {
      controls.update();
    } else {
      if (isTriggerPressed_select) {
        const currentWorldPos = new THREE.Vector3();
        controller1.getWorldPosition(currentWorldPos);
        const currentLocalY = cameraContainer.worldToLocal(currentWorldPos).y;
        // const deltaY = currentLocalY - initialControllerY;
        const deltaY = 0.02
        cameraContainer.position.y += deltaY;
        initialControllerY = currentLocalY;
      }
      if (isTriggerPressed_squeeze) {
        const currentWorldPos = new THREE.Vector3();
        controller1.getWorldPosition(currentWorldPos);
        const currentLocalY = cameraContainer.worldToLocal(currentWorldPos).y;
        // const deltaY = currentLocalY - initialControllerY;
        const deltaY = 0.02
        cameraContainer.position.y -= deltaY;
        initialControllerY = currentLocalY;
      }
    }
    renderer.render(scene, camera);
  }
  renderer.setAnimationLoop(function () {
    animate();
  });
  window.addEventListener("resize", () => {
    if (!renderer.xr.isPresenting) {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth / 1.5, window.innerHeight / 1.5);
    }
  });
  (async () => {
    const mpdUrl = document.getElementById("mpd-input").value;
    const representations = await fetchAndParseMPD(mpdUrl);
    play_control(representations);
  })();
}