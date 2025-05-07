import * as THREE from 'three';
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import camera from "./src/camera";
import scene from "./src/scene";
import { fetchAndParseMPD, play_control} from "./play_control";
import { fetchAndParseMPD_show, play_control_tbb } from "./play_control_show";

export function init3DView(container){
  camera.position.z = 90;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.xr.enabled = true;
  if (!document.querySelector('canvas')) {
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.domElement.style.marginLeft = "15px";
    renderer.domElement.style.marginTop = '4px';
    renderer.setSize(window.innerWidth / 1.5, window.innerHeight / 1.5);
    container.appendChild(renderer.domElement);
  }
  const loader = new DRACOLoader();
  loader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.6/');
  loader.setDecoderConfig({ type: 'js' });
  loader.preload();
  document.getElementById('startButton').addEventListener('click', async function () {
    const mpdUrl = document.getElementById("mpd-input").value;
    if (mpdUrl.includes('Flask_server')) {
      const representations = await fetchAndParseMPD(mpdUrl);
      play_control(representations);
    } else {
      const representations = await fetchAndParseMPD_show(mpdUrl);
      play_control_tbb(representations);
    }
  });
  const controls = new OrbitControls(camera, renderer.domElement);
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