import scene from "./src/scene";
import * as THREE from 'three';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { globalState } from './global';
import camera from "./src/camera";
const loader = new DRACOLoader();
loader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.6/');
loader.setDecoderConfig({ type: 'js' });
loader.preload();


let currentMesh = null;  
let frameCount_fps = 0; 
let lastTime = Date.now();  
let currentFPS = 0;  
function measureFPS() {
    frameCount_fps++;  
    const now = Date.now();
    const timeDelta = now - lastTime;  
    if (timeDelta >= 1000) {
        currentFPS = frameCount_fps;  
        globalState.fps = currentFPS;
        frameCount_fps = 0;  
        lastTime = now;  
    }
    requestAnimationFrame(measureFPS);  
}

function parseMPD(mpdXml) {
    const representations = [];
    const parser = new DOMParser();
    const xmlDoc = parser.parseFromString(mpdXml, "application/xml");
    const adaptationSets = xmlDoc.getElementsByTagName('AdaptationSet');
    let segmentTemplate = null;
    for (const adaptationSet of adaptationSets) {
        segmentTemplate = adaptationSet.querySelector('SegmentTemplate');
    }
    const reps = xmlDoc.getElementsByTagName('Representation');
    for (let i = 0; i < reps.length; i++) {
        const rep = reps[i];
        const bandwidth = parseInt(rep.getAttribute('bandwidth'));
        const size  =  parseInt(rep.getAttribute('size'));
        const id = rep.getAttribute('id');
        const mediaPattern = segmentTemplate ? segmentTemplate.getAttribute('media') : '';
        const qualityFiles = [];
        for (let number = 1450; number <= 1649; number++) {
            const filePath = mediaPattern.replace('$RepresentationID$', id).replace('$Number$', number);
            qualityFiles.push({ bandwidth, size, url: filePath });
        }
        representations.push(qualityFiles);
    }
    console.log("Parsed Representations:", representations);
    return representations;
}

function measureBandwidth() {
    const startTime = Date.now();
    const url = `https://${globalState.IP_address}/drc_show/source/1/redandblack_vox10_1450.drc`;
    return new Promise((resolve, reject) => {
        fetch(url)
            .then(response => response.blob())
            .then(blob => {
                const endTime = Date.now();
                const downloadTime = (endTime - startTime) / 1000;
                const fileSize = blob.size / 1024 / 1024;
                const bandwidth = fileSize / downloadTime;
                resolve(bandwidth);
            })
            .catch(reject);
    });
}

function render(selectedFile) {
    return new Promise((resolve, reject) => {
        loader.load(
            selectedFile,
            function (geometry) {
                geometry.computeVertexNormals();
                geometry.computeBoundingBox();
                // 获取原始包围盒（本地坐标系，未变换前）
                const originalBbox = geometry.boundingBox.clone();
                // console.log('原始本地坐标范围:');
                // console.log('Min:', originalBbox.min);
                // console.log('Max:', originalBbox.max);
                // 计算缩放比例
                const size = new THREE.Vector3();
                originalBbox.getSize(size);
                const maxDimension = Math.max(size.x, size.y, size.z);
                const targetSize = 1.6;
                const scaleFactor = targetSize / maxDimension;
                const material = new THREE.PointsMaterial({
                    vertexColors: true,
                    size: 5 * scaleFactor * 0.5,
                    sizeAttenuation: true,
                    alphaTest: 0.5,
                    transparent: true
                });

                const newMesh = new THREE.Points(geometry, material);
                const center = new THREE.Vector3();
                originalBbox.getCenter(center);
                geometry.translate(-center.x, -center.y, -center.z);

                newMesh.scale.set(scaleFactor, scaleFactor, scaleFactor);
                newMesh.position.set(0, 1.1, -1);

                scene.add(newMesh);
                if (currentMesh) scene.remove(currentMesh);
                currentMesh = newMesh;
                const worldBbox = new THREE.Box3().setFromObject(newMesh);
                // console.log('最终世界坐标范围:');
                // console.log('Min:', worldBbox.min);
                // console.log('Max:', worldBbox.max);
                // 输出变换后的本地坐标范围（可选）
                geometry.computeBoundingBox();
                // console.log('变换后本地坐标范围:');
                // console.log('Min:', geometry.boundingBox.min);
                // console.log('Max:', geometry.boundingBox.max);
                resolve();
            },
            undefined,
            reject
        );
    });
}
export async function fetchAndParseMPD(mpdUrl) {
    const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));
    try {
        const response = await fetch(mpdUrl);
        const mpdXml = await response.text();
        // #############################################################
        // 这里的代码作用是后端处理逻辑的开始的接口
        // globalState.is_loading = 0;//后端处理接口，开始加载
        // globalState.isModalVisible = 1;
        // globalState.showResultButton = 0;
        // const backendUrl = 'http://127.0.0.1:5000';
        // const executionResponse = await fetch(`${backendUrl}/execute-python`, {
        //     method: 'POST',
        //     headers: {
        //         'Content-Type': 'application/json',
        //     },
        // });
        // const executionResult = await executionResponse.json();
        // if (executionResult.status === 'success') {
        //     console.log('Python scripts executed successfully:', executionResult.message);
        // } else {
        //     console.error('Error executing Python scripts:', executionResult.message);
        // }
        // console.log('Waiting 10s before continuing...');
        // await delay(10000); 
        // globalState.is_loading = 1;
        // globalState.isModalVisible = 0;
        // globalState.showResultButton = 1;
        //#############################################################
        const representations = parseMPD(mpdXml);
        return representations;
    }catch(error){
        console.error('Error fetching and parsing MPD:', error);
    }
}



export async function play_control(representations) {
    measureFPS();
    let bandwidth = await measureBandwidth();
    console.log("Initial bandwidth:", bandwidth.toFixed(2), "Mbps");
    let selectedFiles = [];
    let fileCount = 0;
    let frameCounter = 0;
    let inPostPausePhase = false;
    let postPauseFrames = 0;
    let previousBandwidth = bandwidth;  
    if (!representations || representations.length === 0) {
        console.log("No representations available.");
        return;
    }
    let currentQualityLevel = 0;  
    for (let number = 1450; number <= 1749; number++) {
        let selectedFile = null;
        let bandwidthIncrease = ((bandwidth - previousBandwidth) / previousBandwidth) * 100;
        globalState.bandwidthIncrease = bandwidthIncrease;
        if (bandwidthIncrease > 20) {  
            currentQualityLevel = Math.max(currentQualityLevel - 1, 0); 
        } else if (bandwidthIncrease < -20) {  
            currentQualityLevel = Math.min(currentQualityLevel + 1, representations.length - 1); 
        }
        globalState.currentQualityLevel = currentQualityLevel;
        for (let i = currentQualityLevel; i < representations.length; i++) {
            const qualityFiles = representations[i];
            for (let j = 0; j < qualityFiles.length; j++) {
                const rep = qualityFiles[j];
                if (!rep.url.includes(number.toString())) {
                    continue;
                }
                if (bandwidth * 1000000 >= rep.bandwidth) {
                    selectedFile = rep.url;
                    globalState.i = i;
                    try {
                        await render(selectedFile);
                        // await new Promise(resolve => setTimeout(resolve, 600));
                        // 下面的代码的逻辑是视口预测算法的开始接口#####################################
                        // if (!inPostPausePhase) {
                        //     if (frameCounter < 3) {
                        //         console.log(`Frame ${frameCounter + 1} - Camera Position: (${camera.position.x.toFixed(2)}, ${camera.position.y.toFixed(2)}, ${camera.position.z.toFixed(2)})`);
                        //         await logCameraPosition(number, {
                        //             x: camera.position.x,
                        //             y: camera.position.y,
                        //             z: camera.position.z
                        //         });
                        //     }
                        //     frameCounter++;
                        //     if (frameCounter === 3) {
                        //         console.log("播放的帧数量已经达到了3个");
                        //         await new Promise(resolve => setTimeout(resolve, 6000));
                        //         inPostPausePhase = true;  
                        //         postPauseFrames = 0;     
                        //     }
                        // } else {
                        //     postPauseFrames++;
                        //     if (postPauseFrames === 6) {
                        //         inPostPausePhase = false;
                        //         frameCounter = 0;  
                        //     }
                        // }
                        //###########################################################################
                        break;
                    } catch (error) {
                        console.error("Error loading file:", error);
                    }
                    break;
                }
            }
            if (selectedFile) {
                break;
            }
        }
        if (!selectedFile) {
            console.log(`No suitable file found for number ${number} with bandwidth ${bandwidth.toFixed(2)} Mbps/s.`);
        }
        selectedFiles.push({ number, selectedFile });
        fileCount++;
        if (fileCount % 5 === 0) {
            previousBandwidth = bandwidth;
            bandwidth = await measureBandwidth();
            globalState.bandwidth = bandwidth;
        } 
    }
    return selectedFiles;
}

//Min: _Vector3 {x: -0.31202346254006413, y: 0.30000000000000004, z: -1.319061573517628}
//Max: _Vector3 {x: 0.31202346254006413, y: 1.5000000000000001, z: -0.6809384264823718}


