import { globalState } from '../../global.js';

export function createModal() {
    const modal = document.createElement('div');
    modal.classList.add('modal-overlay');
    const modalContent = document.createElement('div');
    modalContent.classList.add('modal-content');
    const closeButton = document.createElement('button');
    closeButton.innerText = '×';
    closeButton.classList.add('modal-close');
    const content = document.createElement('div');
    content.classList.add('modal-body');
    content.style.overflowY = 'auto';
    content.style.maxHeight = '70vh';
    const infoTable = document.createElement('div');
    infoTable.style.display = 'grid';
    infoTable.style.gridTemplateColumns = 'repeat(3, 1fr)';
    infoTable.style.gap = '10px';
    infoTable.style.padding = '10px';
    const fields = {
        frame: createInfoItem('当前帧数'),
        currentPos: createInfoItem('当前坐标'),
        predictPos: createInfoItem('预测坐标'),
        transferBlocks: createInfoItem('传输块数'),
        originBlocks: createInfoItem('原始块数'),
        status: createInfoItem('当前状态')
    };
    Object.values(fields).forEach(item => infoTable.appendChild(item));
    content.appendChild(infoTable);
    modalContent.appendChild(closeButton);
    modalContent.appendChild(content);
    modal.appendChild(modalContent);
    let lastLogId = 0;
    function createInfoItem(label) {
        const container = document.createElement('div');
        const labelElem = document.createElement('strong');
        const valueElem = document.createElement('span');
        labelElem.textContent = label + ': ';
        valueElem.textContent = 'N/A';
        container.appendChild(labelElem);
        container.appendChild(valueElem);
        return container;
    }
    function parseLog(log) {
        const isRecording = log.includes("camera's positions");
        const isPredicting = log.includes("frame's predictions");
        if (isRecording) {
            const frameMatch = log.match(/the (\d+)'s camera's positions/);
            const posMatch = log.match(/positions is \((.*?)\)/);
            return {
                frame: frameMatch[1],
                currentPos: posMatch[1].replace(/,/g, ', '),
                status: 'recording'
            };
        }
        if (isPredicting) {
            const frameMatch = log.match(/the (\d+) frame's predictions/);
            const predictMatch = log.match(/predictions are (.*?) and/);
            const blocksMatch = log.match(/transfer (\d+) blocks with origin block's count is (\d+)/);
            return {
                frame: frameMatch[1],
                predictPos: predictMatch[1],
                transferBlocks: blocksMatch[1],
                originBlocks: blocksMatch[2],
                status: 'predicting'
            };
        }
        return null;
    }
    const fetchLogs = async () => {
        try {
            const backendUrl = 'http://127.0.0.1:5000';
            const response = await fetch(`${backendUrl}/api/logs?since=${lastLogId}`);
            const data = await response.json();
            if (data.logs.length > 0) {
                const logsToProcess = data.logs;
                for (const log of logsToProcess) {
                    const parsed = parseLog(log);
                    console.log("the parsed is:", parsed);
                    if (!parsed) continue;
                    const frameNum = parseInt(parsed.frame, 10);
                    const mod15 = frameNum % 15;
                    const isSpecialFrame = !isNaN(frameNum) && (mod15 >= 6 && mod15 <= 15);
                    if (parsed.frame && (isSpecialFrame)) {
                        if (parsed.frame) fields.frame.lastChild.textContent = parsed.frame;
                        if (parsed.currentPos) fields.currentPos.lastChild.textContent = parsed.currentPos;
                        if (parsed.predictPos) fields.predictPos.lastChild.textContent = 'N/A';
                        if (parsed.transferBlocks) fields.transferBlocks.lastChild.textContent = 'N/A';
                        if (parsed.originBlocks) fields.originBlocks.lastChild.textContent = 'N/A';
                        if (parsed.status) fields.status.lastChild.textContent = 'processing';
                        await new Promise(resolve => setTimeout(resolve, 50)); 
                        continue;
                    }
                    if (parsed.frame) fields.frame.lastChild.textContent = parsed.frame;
                    if (parsed.currentPos) fields.currentPos.lastChild.textContent = parsed.currentPos;
                    if (parsed.predictPos) fields.predictPos.lastChild.textContent = parsed.predictPos;
                    if (parsed.transferBlocks) fields.transferBlocks.lastChild.textContent = parsed.transferBlocks;
                    if (parsed.originBlocks) fields.originBlocks.lastChild.textContent = parsed.originBlocks;
                    if (parsed.status) fields.status.lastChild.textContent = parsed.status;
                    if (parsed.status === 'recording') {
                        fields.predictPos.lastChild.textContent = 'N/A';
                        fields.transferBlocks.lastChild.textContent = 'N/A';
                        fields.originBlocks.lastChild.textContent = 'N/A';
                    }
                    if (parsed.status === 'predicting') {
                        fields.currentPos.lastChild.textContent = 'N/A';
                    }
                    await new Promise(resolve => setTimeout(resolve, 50));
                }
                lastLogId = data.last_id;
                content.scrollTop = content.scrollHeight;
            }
        } catch (error) {
            console.error('日志获取失败:', error);
        }
    };
    const toggleModal = () => {
        const isModalVisible = globalState.isModalVisible;
        modal.style.display = isModalVisible ? 'flex' : 'none';
        if (isModalVisible) {
            fetchLogs();
        }
    };
    const closeModal = () => {
        globalState.isModalVisible = false;
        modal.style.display = 'none';
        Object.values(fields).forEach(item => {
            item.lastChild.textContent = 'N/A';
        });
        fields.status.lastChild.textContent = '';
    };
    closeButton.addEventListener('click', closeModal);
    setInterval(toggleModal, 400);
    return modal;
}