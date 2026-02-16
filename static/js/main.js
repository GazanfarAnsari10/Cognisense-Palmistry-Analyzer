// Global variables
let selectedFile = null;
let isAnalyzing = false;

// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const analyzeBtn = document.getElementById('analyzeBtn');
const retakeBtn = document.getElementById('retakeBtn');
const removeFileBtn = document.getElementById('removeFile');
const loadingSection = document.getElementById('loadingSection');
const uploadSection = document.getElementById('uploadSection');
const progressFill = document.getElementById('progressFill');
const loadingText = document.getElementById('loadingText');
const cameraBtn = document.getElementById('cameraBtn');

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    checkCameraSupport();
});

function initializeEventListeners() {
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Button clicks
    analyzeBtn.addEventListener('click', analyzeImage);
    retakeBtn.addEventListener('click', resetUpload);
    removeFileBtn.addEventListener('click', resetUpload);
    cameraBtn.addEventListener('click', openCamera);
}

function checkCameraSupport() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        cameraBtn.style.display = 'none';
    }
}

// File handling functions
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function processFile(file) {
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
    if (!allowedTypes.includes(file.type)) {
        showError('Please select a valid image file (JPEG, PNG, GIF, or BMP)');
        return;
    }
    
    // Validate file size (max 16MB)
    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File size must be less than 16MB');
        return;
    }
    
    selectedFile = file;
    fileName.textContent = file.name;
    
    // Create preview
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImg.src = e.target.result;
        showImagePreview();
    };
    reader.readAsDataURL(file);
}

function showImagePreview() {
    uploadArea.style.display = 'none';
    imagePreview.style.display = 'block';
    imagePreview.classList.add('fade-in');
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    fileName.textContent = '';
    previewImg.src = '';
    
    uploadArea.style.display = 'block';
    imagePreview.style.display = 'none';
    fileInfo.style.display = 'none';
    
    // Reset any error states
    uploadArea.classList.remove('error-highlight');
}

// Camera functionality
async function openCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        });
        
        showCameraModal(stream);
    } catch (error) {
        console.error('Camera access denied:', error);
        showError('Camera access denied. Please use file upload instead.');
    }
}

function showCameraModal(stream) {
    // Create camera modal
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h3>📸 Take Palm Photo</h3>
                <button class="close-modal" onclick="closeCameraModal()">&times;</button>
            </div>
            <div class="modal-body" style="text-align: center;">
                <video id="cameraVideo" autoplay style="width: 100%; max-width: 400px; border-radius: 10px;"></video>
                <div style="margin-top: 20px;">
                    <button id="captureBtn" class="btn-primary">📷 Capture</button>
                    <button onclick="closeCameraModal()" class="btn-secondary">Cancel</button>
                </div>
                <canvas id="captureCanvas" style="display: none;"></canvas>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    const video = document.getElementById('cameraVideo');
    video.srcObject = stream;
    
    document.getElementById('captureBtn').addEventListener('click', () => {
        capturePhoto(video, stream);
        closeCameraModal();
    });
    
    // Store stream reference for cleanup
    modal.cameraStream = stream;
}

function capturePhoto(video, stream) {
    const canvas = document.getElementById('captureCanvas');
    const context = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    context.drawImage(video, 0, 0);
    
    canvas.toBlob((blob) => {
        const file = new File([blob], 'palm-photo.jpg', { type: 'image/jpeg' });
        processFile(file);
    }, 'image/jpeg', 0.8);
    
    stream.getTracks().forEach(track => track.stop());
}

function closeCameraModal() {
    const modal = document.querySelector('.modal');
    if (modal) {
        if (modal.cameraStream) {
            modal.cameraStream.getTracks().forEach(track => track.stop());
        }
        modal.remove();
    }
}

// Analysis functions
async function analyzeImage() {
    if (!selectedFile || isAnalyzing) return;
    
    isAnalyzing = true;
    showLoadingState();
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Store result in session storage
            sessionStorage.setItem('palmistryAnalysis', JSON.stringify(result));
            
            // Redirect to results page
            window.location.href = '/result';
        } else {
            throw new Error(result.error || 'Analysis failed');
        }
    } catch (error) {
        console.error('Analysis error:', error);
        showError(error.message || 'Failed to analyze image. Please try again.');
        hideLoadingState();
    } finally {
        isAnalyzing = false;
    }
}

function showLoadingState() {
    uploadSection.style.display = 'none';
    loadingSection.style.display = 'block';
    loadingSection.classList.add('fade-in');
    
    // Simulate progress with different stages
    const stages = [
        { progress: 20, text: 'Detecting hand landmarks...' },
        { progress: 40, text: 'Extracting palm lines...' },
        { progress: 60, text: 'Analyzing line characteristics...' },
        { progress: 80, text: 'Generating interpretation...' },
        { progress: 95, text: 'Finalizing results...' }
    ];
    
    let stageIndex = 0;
    const stageInterval = setInterval(() => {
        if (stageIndex < stages.length) {
            const stage = stages[stageIndex];
            progressFill.style.width = stage.progress + '%';
            loadingText.textContent = stage.text;
            stageIndex++;
        } else {
            clearInterval(stageInterval);
        }
    }, 1000);
}

function hideLoadingState() {
    loadingSection.style.display = 'none';
    uploadSection.style.display = 'block';
}

// Utility functions
function showError(message) {
    // Create error notification
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-notification';
    errorDiv.innerHTML = `
        <div class="error-content">
            <span class="error-icon">⚠️</span>
            <span class="error-message">${message}</span>
            <button class="error-close" onclick="this.parentElement.parentElement.remove()">×</button>
        </div>
    `;
    
    // Add error styles if not already present
    if (!document.querySelector('.error-notification-styles')) {
        const styles = document.createElement('style');
        styles.className = 'error-notification-styles';
        styles.textContent = `
            .error-notification {
                position: fixed;
                top: 20px;
                right: 20px;
                background: #fed7d7;
                border: 1px solid #fc8181;
                color: #742a2a;
                padding: 15px;
                border-radius: 10px;
                max-width: 400px;
                z-index: 1000;
                animation: slideInRight 0.3s ease;
            }
            
            .error-content {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .error-close {
                background: none;
                border: none;
                color: #742a2a;
                cursor: pointer;
                font-size: 1.2rem;
                margin-left: auto;
            }
            
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
        document.head.appendChild(styles);
    }
    
    document.body.appendChild(errorDiv);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (errorDiv.parentElement) {
            errorDiv.remove();
        }
    }, 5000);
}

function showSuccess(message) {
    const successDiv = document.createElement('div');
    successDiv.className = 'success-notification';
    successDiv.innerHTML = `
        <div class="success-content">
            <span class="success-icon">✅</span>
            <span class="success-message">${message}</span>
            <button class="success-close" onclick="this.parentElement.parentElement.remove()">×</button>
        </div>
    `;
    
    // Add success styles if not already present
    if (!document.querySelector('.success-notification-styles')) {
        const styles = document.createElement('style');
        styles.className = 'success-notification-styles';
        styles.textContent = `
            .success-notification {
                position: fixed;
                top: 20px;
                right: 20px;
                background: #c6f6d5;
                border: 1px solid #68d391;
                color: #22543d;
                padding: 15px;
                border-radius: 10px;
                max-width: 400px;
                z-index: 1000;
                animation: slideInRight 0.3s ease;
            }
            
            .success-content {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .success-close {
                background: none;
                border: none;
                color: #22543d;
                cursor: pointer;
                font-size: 1.2rem;
                margin-left: auto;
            }
        `;
        document.head.appendChild(styles);
    }
    
    document.body.appendChild(successDiv);
    
    setTimeout(() => {
        if (successDiv.parentElement) {
            successDiv.remove();
        }
    }, 3000);
}

// Validation functions
function validateImage(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = function() {
            // Check minimum dimensions
            if (this.width < 200 || this.height < 200) {
                reject(new Error('Image must be at least 200x200 pixels'));
                return;
            }
            
            // Check aspect ratio (shouldn't be too extreme)
            const aspectRatio = this.width / this.height;
            if (aspectRatio > 3 || aspectRatio < 0.3) {
                reject(new Error('Please use an image with a more standard aspect ratio'));
                return;
            }
            
            resolve(true);
        };
        
        img.onerror = function() {
            reject(new Error('Invalid image file'));
        };
        
        img.src = URL.createObjectURL(file);
    });
}

// Performance monitoring
function trackAnalysisTime() {
    const startTime = performance.now();
    
    return function() {
        const endTime = performance.now();
        const duration = Math.round(endTime - startTime);
        console.log(`Analysis completed in ${duration}ms`);
        
        // Could send to analytics service
        if (window.gtag) {
            window.gtag('event', 'palmistry_analysis', {
                event_category: 'engagement',
                event_label: 'analysis_duration',
                value: duration
            });
        }
    };
}

// Accessibility enhancements
function enhanceAccessibility() {
    // Add ARIA labels
    fileInput.setAttribute('aria-label', 'Select palm image file');
    analyzeBtn.setAttribute('aria-label', 'Analyze palm image');
    
    // Add keyboard navigation
    uploadArea.setAttribute('tabindex', '0');
    uploadArea.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            fileInput.click();
        }
    });
}

// Initialize accessibility on load
document.addEventListener('DOMContentLoaded', enhanceAccessibility);

// Handle page visibility changes
document.addEventListener('visibilitychange', function() {
    if (document.hidden && isAnalyzing) {
        // Page is hidden during analysis - could pause or continue
        console.log('Page hidden during analysis');
    }
});

// Service worker registration for offline functionality
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => console.log('SW registered'))
            .catch(error => console.log('SW registration failed'));
    });
}