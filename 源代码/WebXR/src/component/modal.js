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
    content.innerHTML = '<p>加载中...</p>'; 
    content.style.overflowY = 'auto';
    content.style.maxHeight = '70vh';
    modalContent.appendChild(closeButton);
    modalContent.appendChild(content);
    modal.appendChild(modalContent);
    let lastLogId = 0; 
    const fetchLogs = async () => {
        try {
            const backendUrl = 'http://127.0.0.1:5000';
            const response = await fetch(`${backendUrl}/api/logs?since=${lastLogId}`);
            const data = await response.json();
            if (data.logs.length > 0) {
                data.logs.forEach(log => {
                    const logLine = document.createElement('p');
                    logLine.textContent = log.content;
                    content.appendChild(logLine);
                });
                lastLogId = data.last_id; 
                content.scrollTop = content.scrollHeight;
            }
        } catch (error) {
            console.error('日志获取失败:', error);
        }
    };
    const toggleModal = () => {
        const isModalVisible = globalState.isModalVisible;
        modal.style.display = isModalVisible?'flex' : 'none';
        if (isModalVisible) {
            fetchLogs();
        }
    };
    const closeModal = () => {
        globalState.isModalVisible = false;
        modal.style.display = 'none';
    };
    closeButton.addEventListener('click', closeModal);
    setInterval(toggleModal, 1000); 
    return modal;
}