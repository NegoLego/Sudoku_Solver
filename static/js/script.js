import pyFuncs from './pyFuncs.js';

let fileInput = null;
let dropzone = null;

const titles = [
    'Grayscale Image',
    'Blurred Image',
    'Thresholded, Inverted Image',
    'Contours Identified',
    'Corners Identified',
    'Cut Sudoku Image'
];

window.onload = () => {
    document.getElementById('solve_btn').onclick = uploadImage;

    fileInput = document.getElementById('fileInput');
    dropzone = document.getElementById('dropzone');

    fileInput.addEventListener('change', function() {
        if(this.files && this.files.length > 0) {
            updatePreview(this.files[0]);
        }
    });

    dropzone.addEventListener('click', () => fileInput.click());
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    })
    dropzone.addEventListener('dragover', () => dropzone.classList.add('drag-over'));
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
    dropzone.addEventListener('drop', (e) => {
        dropzone.classList.remove('drag-over');
        fileInput.files = e.dataTransfer.files;
        updatePreview(fileInput.files[0]);
    });
}

function preventDefaults (e) {
    e.preventDefault();
    e.stopPropagation();
}

async function uploadImage() {
    const fileInput = document.getElementById('fileInput');
    const gallery = document.getElementById('gallery');

    if(!fileInput.files[0]) return alert('Please select a file');

    moveLaser();
    // Prepare data
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    let stages = null;
    try {
        // send to python
        const response = await fetch('/solve', {
            method: 'POST',
            body: formData
        });
        stages = await response.json();
        if (stages.error) {
            gallery.innerHTML = `<p style="color:red">Error: ${stages.error}</p>`;
            return;
        }
    } catch (error) {
        console.error(error);
        gallery.innerHTML = '<p style="color:red">Something went wrong!</p>';
    }
    if(stages) fillGallery(stages);
}

function fillGallery(stages) {
    const gallery = document.getElementById('gallery');
    const right_column = document.getElementById('right_column');
    const post_columns = document.getElementById('post_columns');

    if (stages.extract_sudoku.length === 4) {
        post_columns.innerHTML = `<p style="color:red">Error: No Sudoku Found!</p>`;
        showGallery();
        return;
    }

    let right_column_content = `
        <svg class="line_right" width="0">
            <line x1="0" y1="0" x2="100%" y2="100%" stroke="#8174a1" 
                stroke-width="4" stroke-linecap="round" vector-effect="non-scaling-stroke"
                marker-end="url(#arrowhead)" />
        </svg>`;
    for (let i = 0; i < stages.extract_sudoku.length; i++) {
        const stage_string = `
            ${pyFuncHTML()}
            ${line_connectorHTML()}
            <div class="card small">
                <h3>${titles[i]}</h3>
                <img src="${stages.extract_sudoku[i].image}" alt="${titles[i]}" />
                <div class="meta">
                    <span>⏱ ${stages.extract_sudoku[i].time}</span>
                </div>
            </div>
            ${i !== stages.extract_sudoku.length-2 ? arrow_connectorHTML() : line_connectorHTML()}
        `;
        if (i < stages.extract_sudoku.length - 1) right_column_content += stage_string;
        else post_columns.innerHTML = stage_string;
    }
    right_column_content += `
        <svg class="arrow_right" width="0">
            <line x1="100%" y1="0" x2="0" y2="100%"
                  stroke="#8174a1" stroke-width="4"
                  stroke-linecap="round" vector-effect="non-scaling-stroke" 
                  marker-end="url(#arrowhead)"/>
        </svg>
    `;
    right_column.innerHTML = right_column_content;

    post_columns.innerHTML += pyFuncHTML() + line_connectorHTML();

    const nrEmptyCard = document.createElement('div');
    nrEmptyCard.classList.add('card', 'big');
    nrEmptyCard.innerHTML = `<h3>Identify Number/Empty</h3>`;

    let imagesContainer = document.createElement('div');
    imagesContainer.classList.add('images_container');
    for (let i = 0; i < stages.cell_images.images.length; i++) {
        imagesContainer.innerHTML += `
            <div class="cell">
                <div style="color:${chooseColor(stages.cell_images.isNumber[i])}">${stages.cell_images.isNumber[i]}</div>
                <img src=${stages.cell_images.images[i]} alt="Cell Image" />
            </div>
        `;
    }
    nrEmptyCard.appendChild(imagesContainer);
    nrEmptyCard.innerHTML += `
        <div class="meta">
            <span>⏱ ${stages.cell_images.time}</span>
        </div>
    `;
    post_columns.appendChild(nrEmptyCard);

    post_columns.innerHTML += arrow_connectorHTML() + pyFuncHTML() + line_connectorHTML();

    const predictionsCard = document.createElement('div');
    predictionsCard.classList.add('card', 'big');
    predictionsCard.id = 'predictions';
    predictionsCard.innerHTML = `<h3>Model Predictions</h3>`;

    if (Object.keys(stages.predictions).length !== 0) {
        imagesContainer = document.createElement('div');
        imagesContainer.classList.add('images_container');

        for (let i = 0; i < stages.cell_images.images.length; i++) {
            imagesContainer.innerHTML += `
            <div class="cell">
                <div>${stages.predictions.predictions[i]}</div>
                <img src=${stages.cell_images.images[i]} alt="Cell Image" />
            </div>
        `;
        }
        predictionsCard.appendChild(imagesContainer);
        predictionsCard.innerHTML += `
            <div class="meta">
                <span>⏱ ${stages.predictions.time}</span>
            </div>
        `;
    }
    else {
        predictionsCard.innerHTML += '<p>Invalid Sudoku</p>';
        showGallery();
        return;
    }
    post_columns.appendChild(predictionsCard);

    post_columns.innerHTML += arrow_connectorHTML() + pyFuncHTML() + line_connectorHTML();

    const finalCard = document.createElement('div');
    finalCard.classList.add('card', 'big');
    finalCard.innerHTML = `
        <h3>Solved Sudoku</h3>
        <img src=${stages.final_image.image} alt="Solved Sudoku" />
        <div class="meta">
            <span>⏱ ${stages.final_image.time}</span>
        </div>
    `;
    post_columns.appendChild(finalCard);

    // get all buttons and add onclick
    document.querySelectorAll('.pyFunc').forEach((div, i) => {
        let collapsed = true;
        div.innerHTML = 'f(x)';
        div.onclick = () => {
            if (collapsed) {
                div.innerHTML = `<div class="pyCode">
                                    <pre><code>${pyFuncs[i]}</code></pre>
                                 </div>`;
                div.classList.remove('collapsed');
                div.classList.add('expanded');
                collapsed = false;
            }
            else {
                div.innerHTML = 'f(x)';
                div.classList.remove('expanded');
                div.classList.add('collapsed');
                collapsed = true;
            }

        }
    });

    showGallery();
}

function showGallery() {
    const gallery = document.getElementById('gallery');
    const footer = document.getElementById('footer');
    gallery.classList.remove('hidden');
    footer.classList.remove('hidden');
    document.body.classList.remove('no-scroll');
    window.scrollBy({top: 300, left: 0, behavior: 'smooth'});
}

function updatePreview(picture) {
    const preview = document.getElementById('preview');
    preview.classList.remove('hidden');
    preview.src = URL.createObjectURL(picture);
    const placeholder = document.getElementById('placeholder');
    placeholder.classList.add('hidden');
    const instructions = document.getElementById('instructions');
    instructions.classList.add('hidden');
    const button = document.getElementById('solve_btn');
    button.classList.remove('hidden');
}

function chooseColor(value) {
    return value? '#41dd3b' : '#c5381f';
}

function moveLaser() {
    const laser = document.getElementById('laser');
    laser.classList.add('down');
    setTimeout(() => laser.classList.remove('down'), 1000);
}

function pyFuncHTML() {
    return `<div class="pyFunc collapsed"></div>`;
}

function line_connectorHTML() {
    return `
    <svg class="vertical_connector" width="0">
        <line x1="50%" y1="0" x2="50%" y2="100%" stroke="#8174a1" stroke-width="4" stroke-linecap="round"/>
    </svg>
    `;
}

function arrow_connectorHTML() {
    return `
        <svg class="vertical_connector" width="0">
           <line x1="50%" y1="0" x2="50%" y2="100%" stroke="#8174a1" 
               stroke-width="4" stroke-linecap="round" 
               marker-end="url(#arrowhead)" />
        </svg>
    `;
}