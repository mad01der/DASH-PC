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

function startFPSCounter() {
    frameCount_fps = 0;
    lastTime = Date.now();
}

function updateFPSCounter() {
    frameCount_fps++;
    const now = Date.now();
    const timeDelta = now - lastTime;  
    if (timeDelta >= 1000) {
        currentFPS = frameCount_fps;  
        globalState.fps = currentFPS;
        frameCount_fps = 0;  
        lastTime = now;  
    }
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
        for (let number = 1450; number <= 1749; number++) {
            const filePath = mediaPattern.replace('$RepresentationID$', id).replace('$Number$', number);
            qualityFiles.push({ bandwidth, size, url: filePath });
        }
        representations.push(qualityFiles);
    }
    console.log("Parsed Representations:", representations);
    return representations;
}

function measureBandwidth(){
    const startTime = Date.now();
    const url = 'http://localhost/drc_show/source/1/redandblack_vox10_1450.drc';
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
                const material = new THREE.PointsMaterial({
                    vertexColors: true,
                    size: 0.1
                });
                const newMesh = new THREE.Points(geometry, material);
                newMesh.scale.set(0.05, 0.05, 0.05);
                newMesh.position.set(-18, -25, -10);
                scene.add(newMesh);  
                if (currentMesh) {
                    scene.remove(currentMesh);
                }
                currentMesh = newMesh; 
                resolve();
            },
            function (xhr) {
            },
            function (err) {
                console.error(`文件 ${selectedFile} 加载出错:`, err);
                reject(err);
            }
        );
    });
}

export async function fetchAndParseMPD(mpdUrl) {
    try {
        const response = await fetch(mpdUrl);
        const mpdXml = await response.text();
        console.log("the mpdXml is",mpdXml)
        const representations = parseMPD(mpdXml);
        return representations;
    }catch(error){
        console.error('Error fetching and parsing MPD:', error);
    }
}

async function logCameraPosition(frameNumber, position) {
    try {
        const response = await fetch('http://localhost:5000/position-prediction', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                frame: frameNumber,
                x: position.x,
                y: position.y,
                z: position.z
            })
        });
        const result = await response.json();
    } catch (error) {
        console.error('坐标上传失败:', error);
    }
}

export async function play_control(representations) { //TBB-ABR algorithm
    startFPSCounter();
    let bandwidth = await measureBandwidth();
    const FRAME_DURATION = 1 / 8;
    const MAX_BUFFER_SIZE = 5 * FRAME_DURATION * 3; 
    let bufferLevel = MAX_BUFFER_SIZE; 
    let selectedFiles = [];
    let previousBandwidth = bandwidth;  
    if (!representations || representations.length === 0) {
        console.log("No representations available.");
        return selectedFiles;
    }
    let currentQualityLevel = 0;
    for (let chunkStart = 1450; chunkStart <= 1644; chunkStart += 5) {
        globalState.isModalVisible = 1;
        let selectedChunkFiles = [];
        while (bufferLevel <= 0){
            console.log("Buffer depleted, pausing playback to rebuffer...");
            await new Promise(resolve => setTimeout(resolve, 1500)); 
            bufferLevel += MAX_BUFFER_SIZE; 
            bandwidth = await measureBandwidth(); 
            globalState.bufferLevel = bufferLevel.toFixed(2);
        }
        let bandwidthIncrease = ((bandwidth - previousBandwidth) / previousBandwidth) * 100;
        globalState.bandwidthIncrease = bandwidthIncrease;
        if (bandwidthIncrease > 10) {  
            if(bufferLevel > MAX_BUFFER_SIZE / 2){
                currentQualityLevel = Math.max(currentQualityLevel - 1, 0); 
            }
        } else if (bandwidthIncrease < -10) {  
            currentQualityLevel = Math.min(currentQualityLevel + 1, representations.length - 1); 
        }
        globalState.currentQualityLevel = currentQualityLevel;
        try {
            const startTime = performance.now();
            let lastFrameNumber = chunkStart; 
            let chunkId = (chunkStart - 1450)  / 5;
            for (let frameOffset = 0; frameOffset < 5; frameOffset++) {
                const frameNumber = chunkStart + frameOffset;
                if (frameNumber > 1644) break;
                lastFrameNumber = frameNumber;
                const selectedFile = representations[currentQualityLevel].find(
                    file => file.url.includes(frameNumber.toString())
                )?.url;
                if (selectedFile) {
                    await render(selectedFile);
                    updateFPSCounter();
                    if(chunkId % 3 === 0){
                        // await logCameraPosition(frameNumber, {
                        //     x: camera.position.x,
                        //     y: camera.position.y,
                        //     z: camera.position.z
                        // });
                    }
                    selectedChunkFiles.push({
                        number: frameNumber,
                        file: selectedFile
                    });
                }
            }
            const endTime = performance.now();
            const chunkSize = representations[currentQualityLevel]
                .find(file => file.url.includes(lastFrameNumber.toString()))?.size; 

            if (!chunkSize) {
                console.error("Chunk size not found for frame:", lastFrameNumber);
                return; 
            }
            const downloadTime = (chunkSize * 1024 * 8) / (bandwidth * 1e6);
            const renderTime = (endTime - startTime) / 1000; 
            bufferLevel += 5 * FRAME_DURATION - (downloadTime + renderTime);
            bufferLevel = Math.max(0, Math.min(bufferLevel, MAX_BUFFER_SIZE));
            if (chunkStart % (5) === 0) {
                previousBandwidth = bandwidth;
                bandwidth = await measureBandwidth();
                globalState.bandwidth = bandwidth;
            }
            selectedFiles.push({
                chunkStart,
                chunkEnd: chunkStart + 5 - 1,
                files: selectedChunkFiles,
                quality: currentQualityLevel,
                bufferLevel: bufferLevel,
                bandwidth: bandwidth
            });
        } catch (error) {
            console.error("Error loading chunk:", error);
            currentQualityLevel = Math.min(currentQualityLevel + 1, representations.length - 1);
        }
    }
    globalState.isModalVisible = 0;
    return selectedFiles;
}

