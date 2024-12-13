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
  cardContainer.style.gap = '20px';


  function createCard(title, id, initialValue) {
    const card = document.createElement('div');
    card.style.backgroundColor = '#fff';
    card.style.padding = '15px';
    card.style.borderRadius = '8px';
    card.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.1)';
    card.style.textAlign = 'center';

    const cardTitle = document.createElement('h3');
    cardTitle.textContent = title;
    cardTitle.style.margin = '0 0 10px 0';
    cardTitle.style.color = '#555';

    const cardValue = document.createElement('p');
    cardValue.id = id;
    cardValue.textContent = initialValue;
    cardValue.style.fontSize = '18px';
    cardValue.style.color = '#333';

    card.appendChild(cardTitle);
    card.appendChild(cardValue);

    return card;
  }


  cardContainer.appendChild(createCard('Bandwidth', 'bandwidth', '0 Mbps'));
  cardContainer.appendChild(
    createCard('Bandwidth Increase', 'bandwidthIncreaseDisplay', '0%')
  );
  cardContainer.appendChild(
    createCard('Select Quality Level', 'SelectedQualityLevel', '0')
  );
  
  cardContainer.appendChild(
    createCard('FPS', 'fps', '0')
  );
  


  sidebar.appendChild(cardContainer);


  setInterval(updateSideBar, 50); // 每50ms更新一次


  return sidebar;
}


function updateSideBar() {
  const bandwidthIncreaseElement = document.getElementById(
    'bandwidthIncreaseDisplay'
  );
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

  const selectedQualityLevelElement = document.getElementById(
    'SelectedQualityLevel'
  );
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
