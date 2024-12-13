import * as THREE from 'three';
import { OrbitControls } from "three/addons/controls/OrbitControls.js";  // orbit control
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import camera from "./src/camera";
import scene from "./src/scene";
import { fetchAndParseMPD, play_control } from "./play_control";

export function init3DView(container) {

  camera.position.z = 70;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  if (!document.querySelector('canvas')) {  // 检查是否已有 canvas
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.domElement.style.marginLeft = "15px";
    renderer.domElement.style.marginTop = '4px';
    renderer.setSize(window.innerWidth / 1.5, window.innerHeight / 1.5);
    container.appendChild(renderer.domElement);  // 只添加一次
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

  // 渲染场景
  renderer.render(scene, camera);

  // 动画
  function animation() {
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(animation);
  }
  animation();

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth / 1.5, window.innerHeight / 1.5);
  });
}
