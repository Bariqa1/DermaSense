document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const removeBtn = document.getElementById('removeBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const scannerLine = document.getElementById('scannerLine');
    const latencyBadge = document.getElementById('latencyBadge');
    
    const engineStatusDot = document.getElementById('engineStatusDot');
    const engineStatusText = document.getElementById('engineStatusText');

    const emptyState = document.getElementById('emptyState');
    const resultsCard = document.getElementById('resultsCard');
    const printReportBtn = document.getElementById('printReportBtn');

    let currentFile = null;
    let engineDevice = 'CPU';

    // 1. Dynamic Health Check
    async function checkEngineHealth() {
        try {
            engineStatusText.textContent = 'Connecting...';
            const res = await fetch('/api/health');
            const data = await res.json();
            
            if (data.status === 'online') {
                engineDevice = (data.device || 'CPU').toUpperCase();
                engineStatusDot.className = 'status-dot online';
                engineStatusText.textContent = `Engine Active • ${engineDevice}`;
                latencyBadge.textContent = 'Ready';
            } else {
                engineStatusDot.className = 'status-dot offline';
                engineStatusText.textContent = 'Weights Missing';
                latencyBadge.textContent = 'Degraded';
            }
        } catch (err) {
            engineStatusDot.className = 'status-dot offline';
            engineStatusText.textContent = 'Engine Offline';
            latencyBadge.textContent = 'Offline';
        }
    }

    checkEngineHealth();

    // 2. Trigger file chooser
    dropZone.addEventListener('click', () => fileInput.click());

    // Drag & Drop handlers
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('image/')) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            dropZone.style.display = 'none';
            previewContainer.style.display = 'block';
            analyzeBtn.disabled = false;
            latencyBadge.textContent = 'Image Ready';
        };
        reader.readAsDataURL(file);
    }

    // 3. Quick Demo Sample Buttons
    document.querySelectorAll('.btn-sample').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const sampleName = btn.getAttribute('data-sample');
            const sampleUrl = `/static/samples/${sampleName}.jpg`;

            try {
                btn.style.opacity = '0.6';
                const response = await fetch(sampleUrl);
                const blob = await response.blob();
                const file = new File([blob], `${sampleName}_sample.jpg`, { type: 'image/jpeg' });
                
                handleFile(file);
                // Auto trigger inference for instantaneous dynamic experience
                setTimeout(() => {
                    analyzeBtn.click();
                }, 300);
            } catch (err) {
                console.error('Error loading sample image:', err);
            } finally {
                btn.style.opacity = '1';
            }
        });
    });

    // 4. Reset Image
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        currentFile = null;
        fileInput.value = '';
        previewImage.src = '';
        previewContainer.style.display = 'none';
        dropZone.style.display = 'block';
        analyzeBtn.disabled = true;
        latencyBadge.textContent = 'Ready';
        
        // Hide results
        resultsCard.style.display = 'none';
        emptyState.style.display = 'block';
    });

    // 5. Run Dynamic Inference
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        const startTime = performance.now();
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = `
            <svg class="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10" stroke-opacity="0.25"/>
                <path d="M12 2a10 10 0 0 1 10 10" stroke-linecap="round"/>
            </svg>
            Analyzing Dermoscopy Features...
        `;
        scannerLine.style.display = 'block';

        // Update Dynamic Engine Status
        engineStatusDot.className = 'status-dot busy';
        engineStatusText.textContent = 'Processing Lesion...';
        latencyBadge.textContent = 'Inference running...';

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            const elapsed = Math.round(performance.now() - startTime);

            if (result.success && result.data) {
                latencyBadge.textContent = `Latency: ${elapsed}ms`;
                renderResults(result.data);
                
                engineStatusDot.className = 'status-dot online';
                engineStatusText.textContent = `Completed (${elapsed}ms)`;
                setTimeout(() => {
                    engineStatusText.textContent = `Engine Active • ${engineDevice}`;
                }, 3000);
            } else {
                alert('Analysis failed: ' + (result.error || 'Unknown error occurred.'));
                engineStatusDot.className = 'status-dot offline';
                engineStatusText.textContent = 'Analysis Error';
            }
        } catch (err) {
            console.error(err);
            alert('Server error while performing clinical analysis.');
            engineStatusDot.className = 'status-dot offline';
            engineStatusText.textContent = 'Connection Error';
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="5 3 19 12 5 21 5 3"/>
                </svg>
                Run Diagnostic Pipeline
            `;
            scannerLine.style.display = 'none';
        }
    });

    // 6. Render Dynamic Results
    function renderResults(data) {
        emptyState.style.display = 'none';
        resultsCard.style.display = 'block';

        const primary = data.primary;
        document.getElementById('primaryDisease').textContent = primary.disease;
        document.getElementById('primaryConfidence').textContent = `${primary.confidence}%`;
        
        const riskBadge = document.getElementById('riskBadge');
        riskBadge.textContent = primary.metadata.risk || 'Clinical Finding';
        riskBadge.className = `diagnosis-badge badge-${primary.metadata.badge_color || 'info'}`;

        document.getElementById('diseaseDesc').textContent = primary.metadata.description || '';
        document.getElementById('diseaseCategory').textContent = primary.metadata.category || 'Dermatology';

        // Render Differential Diagnosis
        const diffList = document.getElementById('differentialList');
        diffList.innerHTML = '';

        data.differential_diagnosis.forEach((item) => {
            const div = document.createElement('div');
            div.className = 'differential-item';
            div.innerHTML = `
                <div class="differential-info">
                    <span>${item.rank}. ${item.disease}</span>
                    <span style="color: var(--primary-light); font-weight: 600;">${item.confidence}%</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: ${item.confidence}%"></div>
                </div>
            `;
            diffList.appendChild(div);
        });

        // Stage 2: Acne Severity (if applicable)
        const severityBox = document.getElementById('severityBox');
        if (data.is_acne && data.acne_severity) {
            severityBox.style.display = 'block';
            const sev = data.acne_severity;
            
            document.getElementById('severityLabel').textContent = sev.label;
            document.getElementById('severityConfidence').textContent = `${sev.confidence}% confidence`;
            document.getElementById('severityAdvice').textContent = sev.clinical_advice;

            // Highlight active level meter
            for (let i = 0; i <= 3; i++) {
                const step = document.getElementById(`sevStep${i}`);
                if (step) {
                    if (i === sev.level) {
                        step.classList.add('active');
                    } else {
                        step.classList.remove('active');
                    }
                }
            }
        } else {
            severityBox.style.display = 'none';
        }

        // Smooth scroll to results on mobile
        if (window.innerWidth < 960) {
            resultsCard.scrollIntoView({ behavior: 'smooth' });
        }
    }

    // 7. Print Clinical Summary
    if (printReportBtn) {
        printReportBtn.addEventListener('click', () => {
            window.print();
        });
    }
});
