import { globalState } from '../../global'; 

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
  // cardContainer.style.marginTop = '100px';
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
    '显著性检测结果',
    'algorithm1',
    '<button class="result-button">查看结果</button>'
  );
  cardContainer.appendChild(
    createCard('Bandwidth/Bandwidth Increase', 'bandwidth', '0 Mbps', 'Increase: <span id="bandwidthIncreaseDisplay">0%</span>')
  );

  cardContainer.appendChild(
    createCard('Quality Level/FPS', 'SelectedQualityLevel', '0', 'FPS: <span id="fps">0</span>')
  );
  cardContainer.appendChild(algorithm1Card);
  cardContainer.appendChild(createCard('视口预测结果', 'algorithm2', '待定'));
  sidebar.appendChild(cardContainer);
  setInterval(updateSideBar, 50);
  function showModal(imageUrls) {
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
    modalTitle.textContent = '各组显著性检测热力图';
    modalTitle.style.marginBottom = '20px';
    modalTitle.style.color = '#333';
    modalContent.appendChild(modalTitle);
    const imageGrid = document.createElement('div');
    Object.assign(imageGrid.style, {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
      gap: '10px',
      justifyContent: 'center',
      alignItems: 'center'
    });
    imageUrls.forEach((url, index) => {
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
    const closeButton = document.createElement('button');
    closeButton.textContent = '关闭';
    closeButton.style.marginTop = '20px';
    closeButton.style.padding = '10px 20px';
    closeButton.style.backgroundColor = '#007bff';
    closeButton.style.color = 'white';
    //颜色表示为白色
    closeButton.style.border = 'none';
    closeButton.style.borderRadius = '5px';
    closeButton.style.cursor = 'pointer';
    closeButton.addEventListener('click', () => {
      document.body.removeChild(modalOverlay);
    });
    modalContent.appendChild(closeButton);
    modalOverlay.addEventListener('click', (e) => {
      if (e.target === modalOverlay) {
        document.body.removeChild(modalOverlay);
      }
    });

    modalOverlay.appendChild(modalContent);
    document.body.appendChild(modalOverlay);
  }
  
  const resultButton = algorithm1Card.querySelector('.result-button');
  resultButton.addEventListener('click', () => {
    const imageUrls = [];
    const totalImages = 10; 
    for (let i = 1; i <= totalImages; i++) {
      const imageUrl = `/images/render_result_${i}.png`; 
      imageUrls.push(imageUrl);
    }
    showModal(imageUrls);
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
      const selectedQualityLevel = globalState.i;
      selectedQualityLevelElement.innerText = `${selectedQualityLevel + 1}`;
      if (selectedQualityLevel === 0) {
        selectedQualityLevelElement.style.color = 'green';
      } else if (selectedQualityLevel === 1) {
        selectedQualityLevelElement.style.color = '#ffcc00';
      } else if (selectedQualityLevel === 2) {
        selectedQualityLevelElement.style.color = 'red';
      }
    }
  }
  return sidebar;
}


