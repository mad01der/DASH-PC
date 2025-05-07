
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
        const id = rep.getAttribute('id');
        const mediaPattern = segmentTemplate ? segmentTemplate.getAttribute('media') : '';
        const qualityFiles = [];
        for (let number = 1450; number <= 1749; number++) {

            const filePath = mediaPattern.replace('$RepresentationID$', id).replace('$Number$', number);
            qualityFiles.push({ bandwidth, url: filePath });
        }
        representations.push(qualityFiles);
    }
    console.log("Parsed Representations:", representations);
    return representations;
}

async function selectRepresentation(bandwidth, representations) {
    console.log("Current bandwidth:", bandwidth.toFixed(2), "Mbps/s");
    let selectedFiles = [];
    let fileCount = 0;  
    if (!representations || representations.length === 0) {
        console.log("No representations available.");
        return;
    }
    for (let number = 1450; number <= 1749; number++) {
        console.log("File number:", number);
        let selectedFile = null;
        for (let i = 0; i < representations.length; i++) {
            const qualityFiles = representations[i];

            for (let j = 0; j < qualityFiles.length; j++) {
                const rep = qualityFiles[j];

                if (!rep.url.includes(number.toString())) {
                    continue;
                }
                if (bandwidth * 1000000 >= rep.bandwidth) {
                    selectedFile = rep.url;
                    console.log(`Set selected: ${i+1}`);
                    break;
                }
            }
            if (selectedFile) {
                break;
            }
        }
        if (!selectedFile) {
            console.log(`No suitable file found for number ${number} with bandwidth ${bandwidth.toFixed(2)} MB/s.`);
        }
        selectedFiles.push({ number, selectedFile });
        fileCount++;
        if (fileCount % 10 === 0) {
            bandwidth = await measureBandwidth();  
            console.log(`Current bandwidth: ${bandwidth.toFixed(2)} Mbps/s`);
        }
    }
    // console.log("Selected Files:", selectedFiles);
    return selectedFiles;
}

function measureBandwidth() {
    const startTime = Date.now();
    const url = 'source/1/redandblack_vox10_1450.drc';
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

async function adaptStream(mpdUrl) {
    try {
        const response = await fetch(mpdUrl);
        const mpdXml = await response.text();
        const representations = parseMPD(mpdXml);
        const bandwidth = await measureBandwidth();
        selectRepresentation(bandwidth, representations);
    } catch (error) {
        console.error('Error occurred:', error);
    }
}

const mpdUrl = "./pointcloud.mpd";
adaptStream(mpdUrl);
