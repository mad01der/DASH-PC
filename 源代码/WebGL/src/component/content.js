import { ConstantColorFactor } from 'three';
import { init3DView } from '../../main.js';  // 引入 main.js 中的 Three.js 处理逻辑
import { globalState } from '../../global.js';

export function createContent() {
  // 创建主内容容器
  const content = document.createElement('div');
  content.classList.add('content');
  content.style.display = 'flex';
  content.style.flexDirection = 'column';
  content.style.alignItems = 'center';
  content.style.justifyContent = 'center';
  content.style.padding = '20px';
  content.style.backgroundColor = '#f4f4f4';

  const container = document.createElement('div');
  container.classList.add('container');
  container.style.width = '100%';
  container.style.maxWidth = '1200px';  
  container.style.display = 'flex';
  container.style.flexDirection = 'column';
  container.style.alignItems = 'center';


  const description = document.createElement('p');
  description.textContent = 'Explore the 3D point cloud data in the visualization area below.';
  description.style.fontSize = '16px';
  description.style.color = '#555';
  description.style.marginBottom = '30px';

  const threeContainer = document.createElement('div');
threeContainer.id = 'three-container';
threeContainer.style.width = '100%';
threeContainer.style.height = '500px';  
threeContainer.style.marginBottom = '30px';
threeContainer.style.backgroundColor = '#fff'; 
threeContainer.style.borderRadius = '8px';  
threeContainer.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.1)';  
threeContainer.style.position = 'relative';  // 添加相对定位，以便 overlay 在其中定位

window.addEventListener('resize', () => {
  if (threeContainer && threeContainer.clientWidth && threeContainer.clientHeight) {
    init3DView(threeContainer); 
  }
});
const loadingOverlay = document.createElement('div');
loadingOverlay.style.position = 'absolute';
loadingOverlay.style.top = '0';
loadingOverlay.style.left = '0';
loadingOverlay.style.width = '100%';
loadingOverlay.style.height = '100%';
loadingOverlay.style.display = 'none';
loadingOverlay.style.justifyContent = 'center';
loadingOverlay.style.alignItems = 'center';
loadingOverlay.style.backgroundColor = 'rgba(255, 255, 255, 0.8)';
loadingOverlay.style.zIndex = '10';  // 确保它在 threeContainer 之上
loadingOverlay.style.flexDirection = 'column';
const spinner = document.createElement('div');
spinner.style.width = '40px';
spinner.style.height = '40px';
spinner.style.border = '4px solid #f3f3f3';
spinner.style.borderTop = '4px solid #3498db';
spinner.style.borderRadius = '50%';
spinner.style.animation = 'spin 1s linear infinite';

const loadingText = document.createElement('div');
loadingText.textContent = 'Loading...';
loadingText.style.marginTop = '10px';
loadingText.style.color = '#666';
loadingText.style.fontSize = '16px';

const style = document.createElement('style');
style.textContent = `
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;
document.head.appendChild(style);
loadingOverlay.appendChild(spinner);
loadingOverlay.appendChild(loadingText);
threeContainer.appendChild(loadingOverlay);

function updateLoadingState() {
  const isLoading = globalState.is_loading;
  loadingOverlay.style.display = isLoading === 0 ? 'flex' : 'none';
}
container.append(description, threeContainer);
content.appendChild(container);
init3DView(threeContainer);
setInterval(updateLoadingState, 10); 
return content;
}


