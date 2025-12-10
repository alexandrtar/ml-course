// API base URL
const API_BASE = '';

// Check models status on page load
document.addEventListener('DOMContentLoaded', function() {
    checkModelsStatus();
});

// Check available models
async function checkModelsStatus() {
    try {
        const response = await fetch(`${API_BASE}/models`);
        const data = await response.json();
        
        const statusDiv = document.getElementById('modelsStatus');
        statusDiv.innerHTML = `
            <div class="model-status-item ${data.gan_available ? 'model-available' : 'model-unavailable'}">
                GAN Model: ${data.gan_available ? 'Available' : 'Not Available'}
            </div>
            <div class="model-status-item ${data.vae_available ? 'model-available' : 'model-unavailable'}">
                VAE Model: ${data.vae_available ? 'Available' : 'Not Available'}
            </div>
        `;
    } catch (error) {
        document.getElementById('modelsStatus').innerHTML = 
            '<div class="error">Error checking model status</div>';
    }
}

// Generate single faces
document.getElementById('generateForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const requestData = {
        model_type: formData.get('modelType'),
        num_images: parseInt(formData.get('numImages'))
    };
    
    await generateFaces(requestData);
});

// Generate grid
document.getElementById('gridForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const modelType = formData.get('modelType');
    const gridSize = formData.get('gridSize');
    
    await generateGrid(modelType, gridSize);
});

// Generate single faces function
async function generateFaces(requestData) {
    const resultsDiv = document.getElementById('singleResults');
    resultsDiv.innerHTML = '<div class="loading">Generating faces...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        let html = `
            <div class="success">
                Generated ${data.num_images} face(s) using ${data.model.toUpperCase()} model
            </div>
            <div class="image-grid">
        `;
        
        data.images.forEach((imgData, index) => {
            html += `
                <div class="image-item">
                    <img src="${imgData}" alt="Generated face ${index + 1}">
                </div>
            `;
        });
        
        html += '</div>';
        resultsDiv.innerHTML = html;
        
    } catch (error) {
        resultsDiv.innerHTML = `
            <div class="error">
                Error generating faces: ${error.message}
            </div>
        `;
    }
}

// Generate grid function
async function generateGrid(modelType, gridSize) {
    const resultsDiv = document.getElementById('gridResults');
    resultsDiv.innerHTML = '<div class="loading">Generating grid...</div>';
    
    try {
        const response = await fetch(
            `${API_BASE}/generate-grid?model_type=${modelType}&grid_size=${gridSize}`
        );
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        
        resultsDiv.innerHTML = `
            <div class="success">
                Generated ${gridSize}x${gridSize} grid using ${modelType.toUpperCase()} model
            </div>
            <img src="${imageUrl}" alt="Generated grid" class="grid-image">
        `;
        
    } catch (error) {
        resultsDiv.innerHTML = `
            <div class="error">
                Error generating grid: ${error.message}
            </div>
        `;
    }
}

// Reload models function (optional)
async function reloadModels() {
    try {
        const response = await fetch(`${API_BASE}/reload-models`, {
            method: 'POST'
        });
        
        if (response.ok) {
            alert('Models reloaded successfully!');
            checkModelsStatus();
        }
    } catch (error) {
        alert('Error reloading models');
    }
}