import { init3DView } from '../../main.js';  // 引入 main.js 中的 Three.js 处理逻辑

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

  // 创建外部容器，用于包含左侧和右侧的内容
  const container = document.createElement('div');
  container.classList.add('container');
  container.style.width = '100%';
  container.style.maxWidth = '1200px';  // 限制最大宽度
  container.style.display = 'flex';
  container.style.flexDirection = 'column';
  container.style.alignItems = 'center';

  // 创建标题部分
  // const header = document.createElement('h1');
  // header.textContent = 'Point Cloud Visualization';
  // header.style.fontSize = '32px';
  // header.style.color = '#333';
  // header.style.marginBottom = '20px';

  // 创建描述文本
  const description = document.createElement('p');
  description.textContent = 'Explore the 3D point cloud data in the visualization area below.';
  description.style.fontSize = '16px';
  description.style.color = '#555';
  description.style.marginBottom = '30px';

  // 创建用于 Three.js 渲染的容器
  const threeContainer = document.createElement('div');
  threeContainer.id = 'three-container';
  threeContainer.style.width = '100%';
  threeContainer.style.height = '500px';  // 设置渲染区域的高度
  threeContainer.style.marginBottom = '30px';
  threeContainer.style.backgroundColor = '#fff'; // 背景色
  threeContainer.style.borderRadius = '8px';  // 圆角
  threeContainer.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.1)'; // 添加阴影效果

  // 使得 Three.js 渲染内容自适应窗口大小
  window.addEventListener('resize', () => {
    if (threeContainer && threeContainer.clientWidth && threeContainer.clientHeight) {
      init3DView(threeContainer);  // 调用你的 Three.js 渲染函数
    }
  });

  // 将所有内容元素添加到 container 中
  container.append(description, threeContainer);

  // 将 container 添加到 content 中
  content.appendChild(container);

  // 初始化 Three.js 渲染视图，并将渲染器的 canvas 插入到 threeContainer
  init3DView(threeContainer);  // 将渲染器附加到 threeContainer 中

  return content;
}
