import { globalState } from '../../global'; 
import { resultText } from './resultData.js';
export function createSidebar() {
  const sidebar = document.createElement('div');
  sidebar.classList.add('sidebar');

  sidebar.style.padding = '20px';
  sidebar.style.backgroundColor = '#f9f9f9';
  sidebar.style.borderRadius = '8px';
  sidebar.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.1)';
  sidebar.style.fontFamily = 'Arial, sans-serif';
  sidebar.style.width = '300px';

  const welcomeMessage = document.createElement('h1');
  welcomeMessage.textContent = 'Message';
  welcomeMessage.style.textAlign = 'center';
  welcomeMessage.style.marginBottom = '20px';
  welcomeMessage.style.color = '#333';
  sidebar.appendChild(welcomeMessage);
  const cardContainer = document.createElement('div');
  cardContainer.style.display = 'grid';
  cardContainer.style.gridTemplateColumns = '1fr';
  cardContainer.style.gap = '15px';
  function createCard(title, id, initialValue, additionalContent = '') {
    const card = document.createElement('div');
    card.style.backgroundColor = '#fff';
    card.style.padding = '10px';
    card.style.borderRadius = '8px';
    card.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.1)';
    card.style.textAlign = 'center';
    const cardTitle = document.createElement('h3');
    cardTitle.textContent = title;
    cardTitle.style.margin = '0 0 10px 0';
    cardTitle.style.color = '#555';
    const cardValue = document.createElement('p');
    cardValue.id = id;
    cardValue.innerHTML = initialValue; 
    cardValue.style.fontSize = '18px';
    cardValue.style.color = '#333';
    card.appendChild(cardTitle);
    card.appendChild(cardValue);
    if (additionalContent) {
      const additionalInfo = document.createElement('p');
      additionalInfo.innerHTML = additionalContent;
      additionalInfo.style.fontSize = '16px';
      additionalInfo.style.color = '#555';
      card.appendChild(additionalInfo);
    }
    return card;
  }
  const algorithm1Card = createCard(
    '显著性检测结果 & 视口预测算法结果',
    'algorithm1',
    `
      <div style="display: flex; gap: 10px;">
        <button class="result-button" onclick="viewSaliencyResult()">查看显著性检测结果</button>
        <button class="result-button2" onclick="viewViewportPrediction()">查看视口预测算法结果</button>
      </div>
    `
  );
  cardContainer.appendChild(
    createCard('Bandwidth', 'bandwidth', '0 Mbps')
  );
  cardContainer.appendChild(
    createCard('Bandwidth Increase', 'bandwidthIncrease', 
    '<span id="bandwidthIncreaseDisplay">0%</span>')
  );
  cardContainer.appendChild(
    createCard('Quality Level/FPS', 'SelectedQualityLevel', 
    '0', 'FPS: <span id="fps">0</span>')
  );
  cardContainer.appendChild(algorithm1Card);
  sidebar.appendChild(cardContainer);
  setInterval(updateSideBar, 50);
  function showModal(content) {
    const modalOverlay = document.createElement('div');
    Object.assign(modalOverlay.style, {
      position: 'fixed',
      top: '0',
      left: '0',
      width: '100%',
      height: '100%',
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      zIndex: '1000'
    });
  
    const modalContent = document.createElement('div');
    Object.assign(modalContent.style, {
      backgroundColor: 'white',
      padding: '20px',
      borderRadius: '8px',
      boxShadow: '0 4px 8px rgba(0, 0, 0, 0.2)',
      maxWidth: '90%',
      maxHeight: '90vh',
      overflowY: 'auto',
      textAlign: 'center'
    });
  
    const modalTitle = document.createElement('h2');
    modalTitle.textContent = '结果展示';
    modalTitle.style.marginBottom = '20px';
    modalTitle.style.color = '#333';
    modalContent.appendChild(modalTitle);
  
    if (typeof content === 'string') {
      // 显示文本
      const textBlock = document.createElement('pre');
      textBlock.textContent = content;
      textBlock.style.textAlign = 'left';
      textBlock.style.whiteSpace = 'pre-wrap';
      textBlock.style.wordWrap = 'break-word';
      textBlock.style.maxHeight = '70vh';
      textBlock.style.overflowY = 'auto';
      textBlock.style.backgroundColor = '#f7f7f7';
      textBlock.style.padding = '15px';
      textBlock.style.borderRadius = '6px';
      textBlock.style.border = '1px solid #ddd';
      modalContent.appendChild(textBlock);
    } else if (Array.isArray(content)) {
      // 显示图片
      const imageGrid = document.createElement('div');
      Object.assign(imageGrid.style, {
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
        gap: '10px',
        justifyContent: 'center',
        alignItems: 'center'
      });
  
      content.forEach((url, index) => {
        const imgWrapper = document.createElement('div');
        imgWrapper.style.borderRadius = '8px';
        imgWrapper.style.overflow = 'hidden';
        imgWrapper.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.1)';
        imgWrapper.style.position = 'relative';
  
        const img = document.createElement('img');
        img.src = url;
        img.style.width = '100%';
        img.style.height = 'auto';
        img.style.display = 'block';
  
        const labelContainer = document.createElement('div');
        labelContainer.style.textAlign = 'center';
        labelContainer.style.padding = '8px 0';
        labelContainer.style.backgroundColor = 'rgba(255, 255, 255, 0.9)';
  
        const groupLabel = document.createElement('span');
        groupLabel.textContent = `Group ${index + 1}`;
        groupLabel.style.fontSize = '14px';
        groupLabel.style.color = '#333';
        groupLabel.style.fontWeight = 'bold';
  
        labelContainer.appendChild(groupLabel);
        imgWrapper.appendChild(img);
        imgWrapper.appendChild(labelContainer);
        imageGrid.appendChild(imgWrapper);
      });
  
      modalContent.appendChild(imageGrid);
    }
  
    const closeButton = document.createElement('button');
    closeButton.textContent = '关闭';
    Object.assign(closeButton.style, {
      marginTop: '20px',
      padding: '10px 20px',
      backgroundColor: '#007bff',
      color: 'white',
      border: 'none',
      borderRadius: '5px',
      cursor: 'pointer'
    });
    closeButton.addEventListener('click', () => {
      document.body.removeChild(modalOverlay);
    });
  
    modalOverlay.addEventListener('click', (e) => {
      if (e.target === modalOverlay) {
        document.body.removeChild(modalOverlay);
      }
    });
  
    modalContent.appendChild(closeButton);
    modalOverlay.appendChild(modalContent);
    document.body.appendChild(modalOverlay);
  }
  function showTextModal(content) {
    // 创建遮罩层
    const overlay = document.createElement('div');
    overlay.style.position = 'fixed';
    overlay.style.top = 0;
    overlay.style.left = 0;
    overlay.style.width = '100vw';
    overlay.style.height = '100vh';
    overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
    overlay.style.display = 'flex';
    overlay.style.alignItems = 'center';
    overlay.style.justifyContent = 'center';
    overlay.style.zIndex = 1000;
  
    // 创建弹窗容器
    const modal = document.createElement('div');
    modal.style.backgroundColor = '#fff';
    modal.style.padding = '20px';
    modal.style.borderRadius = '10px';
    modal.style.maxWidth = '80%';
    modal.style.maxHeight = '80%';
    modal.style.overflowY = 'auto';
    modal.style.boxShadow = '0 2px 10px rgba(0,0,0,0.3)';
    modal.style.position = 'relative';
  
    // 添加关闭按钮
    const closeButton = document.createElement('button');
    closeButton.textContent = '关闭';
    closeButton.style.position = 'absolute';
    closeButton.style.top = '10px';
    closeButton.style.right = '10px';
    closeButton.style.padding = '5px 10px';
    closeButton.style.border = 'none';
    closeButton.style.backgroundColor = '#f44336';
    closeButton.style.color = '#fff';
    closeButton.style.borderRadius = '5px';
    closeButton.style.cursor = 'pointer';
    closeButton.addEventListener('click', () => {
      document.body.removeChild(overlay);
    });
  
    // 添加文本内容
    const contentElement = document.createElement('pre');
    contentElement.textContent = content;
    contentElement.style.whiteSpace = 'pre-wrap';
    contentElement.style.wordWrap = 'break-word';
  
    modal.appendChild(closeButton);
    modal.appendChild(contentElement);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);
  }
  
  const resultButton = algorithm1Card.querySelector('.result-button');
  resultButton.addEventListener('click', () => {
    const imageUrls = [];
    const totalImages = 24; 
    for (let i = 1; i <= totalImages; i++) {
      const imageUrl = `/images/render_result_${i}.png`; 
      imageUrls.push(imageUrl);
    }
    showModal(imageUrls);
  });
  
  const resultButton2 = algorithm1Card.querySelector('.result-button2');
  resultButton2.addEventListener('click', () => {
    showTextModal(resultText);
  });

  function updateSideBar() {
    const bandwidthIncreaseElement = document.getElementById('bandwidthIncreaseDisplay');
    if (bandwidthIncreaseElement) {
      const bandwidthIncrease = globalState.bandwidthIncrease.toFixed(2);
      bandwidthIncreaseElement.innerText = `${bandwidthIncrease}%`;
      bandwidthIncreaseElement.style.color = bandwidthIncrease > 0 ? 'green' : 'red';
    }
    const bandwidthElement = document.getElementById('bandwidth');
    if (bandwidthElement) {
      bandwidthElement.innerText = `${globalState.bandwidth.toFixed(2)} Mbps`;
    }
    const fps = document.getElementById('fps');
    if (fps) {
      fps.innerText = `${globalState.fps}`;
    }
    const selectedQualityLevelElement = document.getElementById('SelectedQualityLevel');
    if (selectedQualityLevelElement) {
      const selectedQualityLevel = globalState.currentQualityLevel + 1;
      selectedQualityLevelElement.innerText = `${selectedQualityLevel}`;
      if (selectedQualityLevel === 1) {
        selectedQualityLevelElement.style.color = 'green';
      } else if (selectedQualityLevel === 2) {
        selectedQualityLevelElement.style.color = '#ffcc00';
      } else if (selectedQualityLevel === 3) {
        selectedQualityLevelElement.style.color = 'red';
      }
    }
  }
  return sidebar;
}


