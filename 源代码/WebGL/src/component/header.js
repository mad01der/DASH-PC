export function Header() {
    const header = document.createElement('header');
    header.style.backgroundColor = '#505050'; // 深灰色背景
    header.style.color = '#fff'; // 白色字体
    header.style.display = 'flex';
    header.style.justifyContent = 'space-between';
    header.style.alignItems = 'center';
    header.style.padding = '10px 20px';
    header.style.boxSizing = 'border-box';


    const logo = document.createElement('div');
    logo.textContent = 'DASH Point Cloud';
    logo.style.fontSize = '24px';
    logo.style.fontWeight = 'normal';

    const githubLink = document.createElement('a');
    githubLink.href = 'https://github.com/mad01der/DASH-PC';
    githubLink.target = '_blank';
    const githubIcon = document.createElement('img');
    githubIcon.src = './github.png';
    githubIcon.alt = 'GitHub';
    githubIcon.style.width = '30px';
    githubIcon.style.height = '30px';
    githubLink.appendChild(githubIcon);


    header.appendChild(logo);
    header.appendChild(githubLink);


    const controlsBar = document.createElement('div');
    controlsBar.style.display = 'flex';
    controlsBar.style.alignItems = 'center';
    controlsBar.style.justifyContent = 'flex-start';
    controlsBar.style.gap = '10px';
    controlsBar.style.height = '80px';
    controlsBar.style.padding = '10px 20px'; 
    controlsBar.style.boxSizing = 'border-box'; 


    const dropdown = document.createElement('select');
    dropdown.style.padding = '5px';
    dropdown.style.border = '1px solid #ccc';
    dropdown.style.borderRadius = '4px';
    const option = document.createElement('option');
    option.textContent = 'Stream';
    dropdown.appendChild(option);


    const createButton = (text, bgColor, gradient, shadow) => {
        const button = document.createElement('button');
        button.textContent = text;
        button.style.padding = '5px 15px';
        button.style.backgroundColor = bgColor;
        button.style.backgroundImage = gradient;
        button.style.border = 'none';
        button.style.borderRadius = '6px';
        button.style.color = '#fff';
        button.style.cursor = 'pointer';
        button.style.boxShadow = shadow;
        button.style.transition = 'transform 0.2s, box-shadow 0.2s';


        button.addEventListener('mouseenter', () => {
            button.style.transform = 'scale(1.05)';
            button.style.boxShadow = '0px 4px 8px rgba(0, 0, 0, 0.2)';
        });
        button.addEventListener('mouseleave', () => {
            button.style.transform = 'scale(1)';
            button.style.boxShadow = shadow;
        });

        return button;
    };


    const streamButton = createButton(
        'Stream',
        '#007BFF',
        'linear-gradient(145deg, #007bff, #0056b3)',
        '0px 2px 4px rgba(0, 0, 0, 0.2)'
    );


    const input = document.createElement('input');
    input.type = 'text';
    input.placeholder = 'Select MPD file';
    input.id = 'mpd-input'; 
    input.style.flex = '1';
    input.style.padding = '5px';
    input.style.border = '1px solid #ccc';
    input.style.borderRadius = '4px';


    const showOptionsButton = document.createElement('button');
    showOptionsButton.textContent = 'Select MPD';
    showOptionsButton.style.padding = '5px 10px';
    showOptionsButton.style.backgroundColor = '#fff'; 
    showOptionsButton.style.color = '#000000'; 
    showOptionsButton.style.border = '1px solid #ccc';
    showOptionsButton.style.cursor = 'pointer';
    showOptionsButton.style.borderRadius = '4px';


    showOptionsButton.addEventListener('click', () => {
        const overlay = document.createElement('div');
        overlay.style.position = 'fixed';
        overlay.style.top = '0';
        overlay.style.left = '0';
        overlay.style.width = '100%';
        overlay.style.height = '100%';
        overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.4)'; 
        overlay.style.zIndex = '1000';

        const modal = document.createElement('div');
        modal.style.position = 'absolute';
        modal.style.top = '50%';
        modal.style.left = '50%';
        modal.style.transform = 'translate(-50%, -50%)';
        modal.style.backgroundColor = '#fff';
        modal.style.padding = '20px';
        modal.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.2)';
        modal.style.borderRadius = '8px';
        modal.style.width = '300px';
        modal.style.textAlign = 'center';
        modal.style.zIndex = '1010';

        const title = document.createElement('h3');
        title.textContent = 'Select an MPD File';
        title.style.marginBottom = '20px';
        modal.appendChild(title);

        const defaultFile = document.createElement('div');
        defaultFile.id = 'default-file';
        defaultFile.textContent = '展示1(不含视口预测算法)';
        defaultFile.value = 'http://172.22.186.86/drc_show/pointcloud.mpd';
        defaultFile.style.marginBottom = '10px';  
        defaultFile.style.padding = '10px';
        defaultFile.style.border = '1px solid #ccc';
        defaultFile.style.borderRadius = '4px';
        defaultFile.style.cursor = 'pointer';
        defaultFile.style.backgroundColor = '#f9f9f9';
        defaultFile.style.transition = 'background-color 0.3s';
        defaultFile.addEventListener('mouseover', () => (defaultFile.style.backgroundColor = '#e0e0e0'));
        defaultFile.addEventListener('mouseout', () => (defaultFile.style.backgroundColor = '#f9f9f9'));

        // Create second option
        const flaskFile = document.createElement('div');
        flaskFile.id = 'flask-file';
        flaskFile.textContent = '展示2(含有视口预测算法)';
        flaskFile.value = 'http://172.22.186.86/Flask_server/drc_server/pointcloud.mpd';
        flaskFile.style.marginBottom = '20px';  
        flaskFile.style.padding = '10px';
        flaskFile.style.border = '1px solid #ccc';
        flaskFile.style.borderRadius = '4px';
        flaskFile.style.cursor = 'pointer';
        flaskFile.style.backgroundColor = '#f9f9f9';
        flaskFile.style.transition = 'background-color 0.3s';
        flaskFile.addEventListener('mouseover', () => (flaskFile.style.backgroundColor = '#e0e0e0'));
        flaskFile.addEventListener('mouseout', () => (flaskFile.style.backgroundColor = '#f9f9f9'));

        // Add both to modal
        modal.appendChild(defaultFile);
        modal.appendChild(flaskFile);

        const confirmButton = document.createElement('button');
        confirmButton.textContent = 'Confirm';
        confirmButton.style.padding = '10px 20px';
        confirmButton.style.backgroundColor = '#007BFF';
        confirmButton.style.color = '#fff';
        confirmButton.style.border = 'none';
        confirmButton.style.borderRadius = '4px';
        confirmButton.style.cursor = 'pointer';
        modal.appendChild(confirmButton);

        const cancelButton = document.createElement('button');
        cancelButton.textContent = 'Cancel';
        cancelButton.style.marginLeft = '10px';
        cancelButton.style.padding = '10px 20px';
        cancelButton.style.backgroundColor = '#ccc';
        cancelButton.style.color = '#000';
        cancelButton.style.border = 'none';
        cancelButton.style.borderRadius = '4px';
        cancelButton.style.cursor = 'pointer';
        modal.appendChild(cancelButton);

        document.body.appendChild(overlay);
        document.body.appendChild(modal);

        confirmButton.addEventListener('click', () => {
            document.body.removeChild(overlay);
            document.body.removeChild(modal);
        });
        defaultFile.addEventListener('click', () => {
            document.getElementById('mpd-input').value = defaultFile.value;
            document.body.removeChild(overlay);
            document.body.removeChild(modal);
        });
        flaskFile.addEventListener('click', () => {
            document.getElementById('mpd-input').value = flaskFile.value;
            document.body.removeChild(overlay);
            document.body.removeChild(modal);
        });
        cancelButton.addEventListener('click', () => {
            document.getElementById('mpd-input').value = null;
            document.body.removeChild(overlay);
            document.body.removeChild(modal);
        });

        overlay.addEventListener('click', () => {
            document.body.removeChild(overlay);
            document.body.removeChild(modal);
        });
    });


    controlsBar.appendChild(input);
    controlsBar.appendChild(showOptionsButton);

    const stopButton = document.createElement('button');
    stopButton.textContent = 'Refresh';
    stopButton.style.padding = '5px 10px';
    stopButton.style.backgroundColor = '#fff';
    stopButton.style.border = '1px solid #ccc';
    stopButton.style.cursor = 'pointer';
    stopButton.addEventListener('click', () => {
        location.reload();
    });


    const loadButton = createButton(
        'Load',
        '#007BFF',
        'linear-gradient(145deg, #007bff, #0056b3)',
        '0px 2px 4px rgba(0, 0, 0, 0.2)'
    );
    loadButton.id = "startButton";
    loadButton.addEventListener('click', () => {
        const inputValue = document.getElementById('mpd-input').value; // 获取 mpd-select 的值
        const event = new CustomEvent('inputValueSelected', {
            detail: { value: inputValue }
        });
        window.dispatchEvent(event); // 触发事件，传递值
    });
    
    controlsBar.appendChild(streamButton);
    controlsBar.appendChild(input);
    controlsBar.appendChild(showOptionsButton);
    controlsBar.appendChild(stopButton);
    controlsBar.appendChild(loadButton);
    const container = document.createElement('div');
    container.appendChild(header); // 顶部部分
    container.appendChild(controlsBar); // 控件部分
    return container;
}
