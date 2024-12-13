import scene from "./src/scene";
import * as THREE from 'three';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { globalState } from './global';
const loader = new DRACOLoader();
loader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.6/');
loader.setDecoderConfig({ type: 'js' });
loader.preload();


let currentMesh = null;  
let frameCount = 0;  // 帧计数
let lastTime = Date.now();  // 上次更新时间
let currentFPS = 0;  // 当前 FPS

function measureFPS() {
    frameCount++;  // 每帧递增

    const now = Date.now();
    const timeDelta = now - lastTime;  // 与上次计算的时间差

    // 每秒钟计算一次 FPS
    if (timeDelta >= 1000) {
        currentFPS = frameCount;  // 设置当前帧率为计算得出的帧数
        console.log(`FPS: ${currentFPS}`);
        globalState.fps = currentFPS;
        frameCount = 0;  // 重置帧计数
        lastTime = now;  // 更新上次更新时间
    }

    requestAnimationFrame(measureFPS);  // 请求下一帧的渲染，继续统计
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

function measureBandwidth() {
    const startTime = Date.now();
    const url = 'http://localhost/drc/source/1/redandblack_vox10_1450.drc';
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
                scene.add(newMesh);  // 添加新点云
                if (currentMesh) {
                    scene.remove(currentMesh);
                }
                currentMesh = newMesh; 
                resolve();
            },
            function (xhr) {
                // console.log(`文件 ${selectedFile} 加载中: ${(xhr.loaded / xhr.total) * 100}%`);
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
        // 获取MPD文件
        const response = await fetch(mpdUrl);
        const mpdXml = await response.text();
        
        // 调用parseMPD来解析获取到的XML内容
        const representations = parseMPD(mpdXml);
        
        return representations;
    } catch (error) {
        console.error('Error fetching and parsing MPD:', error);
    }
}

export async function play_control(representations) {

    measureFPS();

    
    let bandwidth = await measureBandwidth();
    console.log("Initial bandwidth:", bandwidth.toFixed(2), "Mbps");

    let selectedFiles = [];
    let fileCount = 0;

    let previousBandwidth = bandwidth;  // 初始化上一次带宽为当前带宽

    if (!representations || representations.length === 0) {
        console.log("No representations available.");
        return;
    }

    // BBA 初始化阶段
    let currentQualityLevel = 0;  
    let previousQualityLevel = -1; 

    for (let number = 1450; number <= 1749; number++) {
        console.log("File number:", number);
        let selectedFile = null;

        // 计算当前带宽和上一次带宽的增幅百分比
        let bandwidthIncrease = ((bandwidth - previousBandwidth) / previousBandwidth) * 100;
        console.log("Bandwidth increase percentage:", bandwidthIncrease.toFixed(2), "%");
        globalState.bandwidthIncrease = bandwidthIncrease;
        // 根据增幅调整质量级别
        if (bandwidthIncrease > 20) {  
            console.log("Bandwidth increased significantly, increasing quality.");
            currentQualityLevel = Math.max(currentQualityLevel - 1, 0);  // 降低质量级别
            console.log("currentQualityLevel",currentQualityLevel + 1);
        } else if (bandwidthIncrease < -20) {  
            console.log("Bandwidth decreased significantly, decreasing quality.");
            currentQualityLevel = Math.min(currentQualityLevel + 1, representations.length - 1);  // 增加质量级别
            console.log("currentQualityLevel",currentQualityLevel + 1);
        }
        globalState.currentQualityLevel = currentQualityLevel;
        // 根据当前质量级别选择合适的文件
        for (let i = currentQualityLevel; i < representations.length; i++) {
            const qualityFiles = representations[i];
            for (let j = 0; j < qualityFiles.length; j++) {
                const rep = qualityFiles[j];

                // 检查文件编号是否匹配
                if (!rep.url.includes(number.toString())) {
                    continue;
                }

                // 检查带宽是否足够
                if (bandwidth * 1000000 >= rep.bandwidth) {
                    selectedFile = rep.url;
                    console.log(`Selected quality level: ${i + 1}`);
                    globalState.i = i;
                    try {
                        await render(selectedFile);  // 渲染选中的文件
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

        // 每 5 个文件重新测量带宽
        if (fileCount % 5 === 0) {
            previousBandwidth = bandwidth;  // 更新上一次带宽
            bandwidth = await measureBandwidth();  // 重新测量带宽
            globalState.bandwidth = bandwidth;
            console.log(`Current bandwidth: ${bandwidth.toFixed(2)} Mbps`);
        }
    }

    return selectedFiles;
}




