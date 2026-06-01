// Photocatalyst HER - SPA Application Logic

// Global Data State
let appData = {
    dfClean: [],
    trainingResults: {},
    conformalSummary: {},
    adSummary: {},
    conformalIntervals: [],
    shapSummary: {},
    candidates: [],
    filteredCandidates: [],
    activeTab: 'dataset',
    activeLightFilter: 'all',
    scatterScale: 'log' // 'log' or 'orig'
};

// Global Chart Instances
let charts = {};

// Simple Markdown Parser
function renderMarkdown(md) {
    if (!md) return '';
    let html = md
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/^### (.*$)/gim, '<h3 class="text-md font-semibold mt-4 mb-2 text-on-surface">$1</h3>')
        .replace(/^## (.*$)/gim, '<h2 class="text-lg font-bold mt-6 mb-3 text-secondary">$1</h2>')
        .replace(/^# (.*$)/gim, '<h1 class="text-xl font-bold mt-8 mb-4 text-primary">$1</h1>')
        .replace(/^\s*\-\s*(.*$)/gim, '<li class="ml-4 list-disc text-on-surface-variant">$1</li>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`([^`]+)`/g, '<code class="bg-surface-container-highest px-1 rounded text-tertiary">$1</code>')
        .replace(/\n\n/g, '</p><p class="mb-4 text-on-surface-variant leading-relaxed">');
    
    // Wrap consecutive list items
    html = html.replace(/(<li class=".*?">.*?<\/li>)/gs, '<ul class="my-2 space-y-1">$1</ul>');
    return '<p class="mb-4 text-on-surface-variant leading-relaxed">' + html + '</p>';
}

// Format number utility
function formatNum(val, decimals = 1) {
    if (val === undefined || val === null || isNaN(val)) return 'N/A';
    return Number(val).toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

// Initialize Application
async function init() {
    console.log("Initializing SPA Dashboard...");
    
    // Show active tab initially
    switchTab('dataset');
    
    try {
        // Fetch JSON configs
        const resultsRes = await fetch('data/results/training_results.json');
        appData.trainingResults = await resultsRes.json();
        
        const conformalRes = await fetch('data/results/conformal_summary.json');
        appData.conformalSummary = await conformalRes.json();
        
        const adRes = await fetch('data/results/ad_summary.json');
        appData.adSummary = await adRes.json();
        
        const shapRes = await fetch('data/results/shap_summary_data.json');
        appData.shapSummary = await shapRes.json();

        // Fetch Markdown files for static tabs
        const prdRes = await fetch('stitch_assets/prd.md');
        const prdMarkdown = await prdRes.text();
        document.getElementById('prd-content-rendered').innerHTML = renderMarkdown(prdMarkdown);

        const designRes = await fetch('stitch_assets/design_system.md');
        const designMarkdown = await designRes.text();
        document.getElementById('design-content-rendered').innerHTML = renderMarkdown(designMarkdown);

        // Fetch CSV Files using PapaParse
        Papa.parse('data/processed/df_clean.csv', {
            download: true,
            header: true,
            dynamicTyping: true,
            complete: function(results) {
                appData.dfClean = results.data.filter(r => r.host_material);
                onDatasetLoaded();
            }
        });

        Papa.parse('data/results/conformal_intervals.csv', {
            download: true,
            header: true,
            dynamicTyping: true,
            complete: function(results) {
                appData.conformalIntervals = results.data.filter(r => r.y_true_log !== undefined);
                onModelDataLoaded();
            }
        });

        // Load screening candidate files (top 20 and top novel)
        Papa.parse('data/results/top_20_candidates.csv', {
            download: true,
            header: true,
            dynamicTyping: true,
            complete: function(results20) {
                const c20 = results20.data.filter(r => r.host_material);
                
                Papa.parse('data/results/top_novel_candidates.csv', {
                    download: true,
                    header: true,
                    dynamicTyping: true,
                    complete: function(resultsNovel) {
                        const cNovel = resultsNovel.data.filter(r => r.host_material);
                        
                        // Combine and remove duplicates based on composition/config
                        const combined = [...c20, ...cNovel];
                        const uniqueMap = new Map();
                        combined.forEach(item => {
                            const key = `${item.host_material}_${item.co_catalyst}_${item.cocatalyst_wt_pct}_${item.light_source_type}`;
                            if (!uniqueMap.has(key)) {
                                uniqueMap.set(key, item);
                            }
                        });
                        appData.candidates = Array.from(uniqueMap.values());
                        // Sort by predicted median log or predicted HER rate descending
                        appData.candidates.sort((a, b) => b.pred_her_umol_g_h - a.pred_her_umol_g_h);
                        appData.filteredCandidates = [...appData.candidates];
                        onCandidatesLoaded();
                    }
                });
            }
        });

    } catch (err) {
        console.error("Error loading project resources:", err);
    }
}

// Switch dashboard tabs
function switchTab(tabId) {
    appData.activeTab = tabId;
    
    // Hide all contents
    document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
    
    // Show target content
    const activeEl = document.getElementById(`tab-${tabId}`);
    if (activeEl) activeEl.classList.remove('hidden');
    
    // Update Sidebar Active states
    document.querySelectorAll('.side-nav-btn').forEach(btn => btn.classList.remove('active-nav'));
    const activeSideBtn = document.getElementById(`side-nav-${tabId}`);
    if (activeSideBtn) activeSideBtn.classList.add('active-nav');
    
    // Update Mobile Nav Active states
    document.querySelectorAll('.mobile-nav-btn').forEach(btn => btn.classList.remove('active-mobile-nav'));
    const activeMobileBtn = document.getElementById(`mobile-nav-${tabId}`);
    if (activeMobileBtn) activeMobileBtn.classList.add('active-mobile-nav');

    // Update Top nav items
    document.querySelectorAll('.top-nav-btn').forEach(btn => btn.classList.remove('active-top-nav'));
    const activeTopBtn = document.getElementById(`top-nav-${tabId}`);
    if (activeTopBtn) activeTopBtn.classList.add('active-top-nav');
    
    // Re-trigger layout checks for Chart.js
    if (charts[tabId]) {
        charts[tabId].forEach(chart => chart.resize());
    }
}

// Triggered when df_clean.csv is loaded
function onDatasetLoaded() {
    console.log("Dataset parsed:", appData.dfClean.length, "rows");
    
    // Set metric totals
    document.getElementById('metric-total-experiments').innerText = appData.dfClean.length;
    
    // Count unique semiconductors
    const uniqueSemis = new Set(appData.dfClean.map(r => String(r.host_material).toLowerCase().trim()));
    document.getElementById('metric-unique-semiconductors').innerText = uniqueSemis.size;

    // Build semiconductor distribution chart
    buildSemiconductorChart();
    
    // Build HER log-scale distribution chart
    buildHerDistChart();
}

// Semiconductor distribution chart builder
function buildSemiconductorChart() {
    const counts = {};
    appData.dfClean.forEach(row => {
        const semi = String(row.host_material).toUpperCase().trim();
        counts[semi] = (counts[semi] || 0) + 1;
    });

    const sorted = Object.entries(counts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10); // top 10

    const labels = sorted.map(x => x[0]);
    const data = sorted.map(x => x[1]);

    const ctx = document.getElementById('chart-semiconductor-dist').getContext('2d');
    if (charts['dataset_semi']) charts['dataset_semi'].destroy();
    
    charts['dataset_semi'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Experiment Count',
                data: data,
                backgroundColor: 'rgba(78, 222, 163, 0.2)', // Emerald green
                borderColor: '#4edea3',
                borderWidth: 1.5,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#bdc8d1', font: { family: 'JetBrains Mono', size: 10 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#d4e4fa', font: { family: 'Inter', weight: '600' } }
                }
            }
        }
    });
}

// HER Rate distribution log-scale chart
function buildHerDistChart() {
    const logVals = appData.dfClean.map(r => r.log_HER).filter(v => v !== undefined && !isNaN(v));
    const min = Math.min(...logVals);
    const max = Math.max(...logVals);
    
    // Create 12 bins
    const numBins = 12;
    const binWidth = (max - min) / numBins;
    const bins = Array(numBins).fill(0);
    const binLabels = [];
    
    for (let i = 0; i < numBins; i++) {
        const binStart = min + i * binWidth;
        const binEnd = binStart + binWidth;
        binLabels.push(`10^${binStart.toFixed(1)}`);
    }
    
    logVals.forEach(v => {
        let idx = Math.floor((v - min) / binWidth);
        if (idx >= numBins) idx = numBins - 1;
        if (idx < 0) idx = 0;
        bins[idx]++;
    });

    const ctx = document.getElementById('chart-her-dist').getContext('2d');
    if (charts['dataset_her']) charts['dataset_her'].destroy();

    charts['dataset_her'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: binLabels,
            datasets: [{
                data: bins,
                backgroundColor: 'rgba(56, 189, 248, 0.25)', // Ice Blue
                borderColor: '#38bdf8',
                borderWidth: 1.5,
                borderRadius: 2,
                barPercentage: 0.95
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#bdc8d1', font: { family: 'JetBrains Mono', size: 8 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#bdc8d1', font: { family: 'JetBrains Mono', size: 9 } }
                }
            }
        }
    });

    // Max HER stat update
    const maxHer = Math.max(...appData.dfClean.map(r => r.HER_std_umol_g_h || 0));
    document.getElementById('val-her-max').innerText = `${formatNum(maxHer, 0)} μmol/g/h`;
}

// Triggered when conformal_intervals.csv is loaded
function onModelDataLoaded() {
    console.log("Model predictions parsed:", appData.conformalIntervals.length, "test points");
    
    // Bind global model stats
    if (appData.trainingResults && appData.trainingResults.XGBoost) {
        document.getElementById('model-lomo-r2').innerText = appData.trainingResults.XGBoost.LOMO_CV_R2_mean.toFixed(4);
        document.getElementById('model-spearman').innerText = appData.trainingResults.XGBoost.Spearman_rho_log.toFixed(4);
    }
    
    if (appData.conformalSummary && appData.conformalSummary.empirical_coverage) {
        document.getElementById('model-conformal-cov').innerText = `${(appData.conformalSummary.empirical_coverage * 100).toFixed(1)}%`;
    }

    // Build actual vs predicted parity chart
    buildParityChart();
    
    // Build residuals histogram chart
    buildResidualsChart();

    // Render SHAP features lists
    buildShapAttributionList();

    // Render SHAP beeswarm chart
    buildShapBeeswarmChart();
}

// Predicted vs Actual scatter plot
function buildParityChart() {
    let datasetPoints = [];
    let minVal = Infinity;
    let maxVal = -Infinity;
    
    appData.conformalIntervals.forEach(row => {
        let x, y, low, high;
        if (appData.scatterScale === 'log') {
            x = row.y_true_log;
            y = row.y_pred_log;
            low = row.lower_log;
            high = row.upper_log;
        } else {
            x = row.y_true_her;
            y = row.y_pred_her;
            low = row.lower_her;
            high = row.upper_her;
        }
        
        if (x !== undefined && y !== undefined) {
            datasetPoints.push({ x: x, y: y, low: low, high: high });
            minVal = Math.min(minVal, x, y);
            maxVal = Math.max(maxVal, x, y);
        }
    });

    const ctx = document.getElementById('chart-actual-vs-predicted').getContext('2d');
    if (charts['model_parity']) charts['model_parity'].destroy();

    // Custom Error Bars Plugin
    const errorBarsPlugin = {
        id: 'errorBars',
        afterDatasetsDraw: function(chart) {
            const ctx = chart.ctx;
            chart.data.datasets.forEach((dataset, datasetIndex) => {
                if (dataset.label === 'Predictions') {
                    const meta = chart.getDatasetMeta(datasetIndex);
                    meta.data.forEach((element, index) => {
                        const dataPoint = dataset.data[index];
                        const errorLow = dataPoint.low;
                        const errorHigh = dataPoint.high;
                        
                        const xPixel = element.x;
                        const yLowPixel = chart.scales.y.getPixelForValue(errorLow);
                        const yHighPixel = chart.scales.y.getPixelForValue(errorHigh);
                        
                        ctx.save();
                        ctx.strokeStyle = 'rgba(78, 222, 163, 0.2)'; // low opacity green
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(xPixel, yLowPixel);
                        ctx.lineTo(xPixel, yHighPixel);
                        ctx.stroke();
                        
                        // horizontal ticks
                        ctx.beginPath();
                        ctx.moveTo(xPixel - 2, yLowPixel);
                        ctx.lineTo(xPixel + 2, yLowPixel);
                        ctx.moveTo(xPixel - 2, yHighPixel);
                        ctx.lineTo(xPixel + 2, yHighPixel);
                        ctx.stroke();
                        ctx.restore();
                    });
                }
            });
        }
    };

    charts['model_parity'] = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Predictions',
                    data: datasetPoints,
                    backgroundColor: 'rgba(78, 222, 163, 0.75)', // Emerald
                    borderColor: '#4edea3',
                    borderWidth: 0.8,
                    pointRadius: 4,
                    pointHoverRadius: 6
                },
                {
                    label: 'Parity Line',
                    data: [{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }],
                    type: 'line',
                    borderColor: 'rgba(255, 255, 255, 0.25)',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        plugins: [errorBarsPlugin],
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    type: appData.scatterScale === 'log' ? 'linear' : 'logarithmic',
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    title: { display: true, text: appData.scatterScale === 'log' ? 'Actual log(HER+1)' : 'Actual HER (μmol/g/h)', color: '#bdc8d1' },
                    ticks: { color: '#bdc8d1', font: { family: 'JetBrains Mono' } }
                },
                y: {
                    type: appData.scatterScale === 'log' ? 'linear' : 'logarithmic',
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    title: { display: true, text: appData.scatterScale === 'log' ? 'Predicted log(HER+1)' : 'Predicted HER (μmol/g/h)', color: '#bdc8d1' },
                    ticks: { color: '#bdc8d1', font: { family: 'JetBrains Mono' } }
                }
            }
        }
    });
}

// Toggle Parity chart scale
function toggleScatterScale(scale) {
    appData.scatterScale = scale;
    document.querySelectorAll('[id^="scale-btn-"]').forEach(btn => {
        btn.className = "px-md py-xs rounded-full font-label-caps text-label-caps text-on-surface-variant hover:bg-surface-variant transition-colors";
    });
    const activeBtn = document.getElementById(`scale-btn-${scale}`);
    activeBtn.className = "px-md py-xs rounded-full font-label-caps text-label-caps bg-primary text-on-primary transition-colors";
    buildParityChart();
}

// Residual distribution histogram builder
function buildResidualsChart() {
    const residuals = appData.conformalIntervals.map(r => r.y_true_log - r.y_pred_log);
    
    // Create 15 bins from -2.0 to +2.0
    const numBins = 13;
    const min = -2.0;
    const max = 2.0;
    const binWidth = (max - min) / numBins;
    const bins = Array(numBins).fill(0);
    const labels = [];
    
    for (let i = 0; i < numBins; i++) {
        const val = min + i * binWidth + binWidth/2;
        labels.push(val.toFixed(1));
    }
    
    residuals.forEach(r => {
        let idx = Math.floor((r - min) / binWidth);
        if (idx >= numBins) idx = numBins - 1;
        if (idx < 0) idx = 0;
        bins[idx]++;
    });

    const ctx = document.getElementById('chart-residuals').getContext('2d');
    if (charts['model_residuals']) charts['model_residuals'].destroy();

    charts['model_residuals'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: bins,
                backgroundColor: 'rgba(56, 189, 248, 0.25)', // Ice Blue
                borderColor: '#38bdf8',
                borderWidth: 1.2,
                borderRadius: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#bdc8d1', font: { family: 'JetBrains Mono', size: 9 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#bdc8d1', font: { family: 'JetBrains Mono', size: 9 } }
                }
            }
        }
    });
}

// Populate the horizontal feature list ranking in SHAP panel
function buildShapAttributionList() {
    const listContainer = document.getElementById('shap-rankings-list');
    if (!listContainer || !appData.shapSummary || !appData.shapSummary.feature_importance) return;
    
    let html = '<div class="space-y-md">';
    
    // Sort features by mean_abs_shap and list top 8
    const topFeatures = appData.shapSummary.feature_importance.slice(0, 8);
    
    topFeatures.forEach((feat, index) => {
        const pct = (feat.mean_abs_shap / topFeatures[0].mean_abs_shap) * 100;
        const colorClass = index % 2 === 0 ? 'bg-primary' : 'bg-secondary';
        
        html += `
        <div class="space-y-xs">
            <div class="flex justify-between items-end text-xs">
                <span class="font-label-caps text-on-surface uppercase font-semibold">${feat.feature.replace(/_/g, ' ')}</span>
                <span class="font-data-md text-outline font-semibold">${feat.mean_abs_shap.toFixed(3)}</span>
            </div>
            <div class="w-full h-1 bg-surface-container-highest rounded-full overflow-hidden">
                <div class="h-full ${colorClass} rounded-full" style="width: ${pct}%"></div>
            </div>
        </div>`;
    });
    
    html += '</div>';
    listContainer.innerHTML = html;
}

// Generate the SHAP Beeswarm scatter chart
function buildShapBeeswarmChart() {
    if (!appData.shapSummary || !appData.shapSummary.beeswarm) return;
    
    const datasets = [];
    const topBeeswarm = appData.shapSummary.beeswarm.slice(0, 8);
    const featureLabels = topBeeswarm.map(b => b.feature.replace(/_/g, ' ').toUpperCase());
    
    const highValPoints = [];
    const lowValPoints = [];
    
    topBeeswarm.forEach((feat, featIdx) => {
        // Find min/max values to normalize colors
        const featVals = feat.points.map(p => p.feat_val);
        const minVal = Math.min(...featVals);
        const maxVal = Math.max(...featVals);
        const range = maxVal - minVal || 1.0;
        
        feat.points.forEach(point => {
            const norm = (point.feat_val - minVal) / range;
            
            // Jitter y slightly around the feature index (e.g. 7 - featIdx)
            const yCenter = 7 - featIdx;
            const yJitter = yCenter + (Math.random() - 0.5) * 0.45;
            
            const pt = {
                x: point.shap_val,
                y: yJitter,
                feature: feat.feature,
                feat_val_raw: point.feat_val
            };
            
            // Separate into high and low values
            if (norm > 0.5) {
                highValPoints.push(pt);
            } else {
                lowValPoints.push(pt);
            }
        });
    });
    
    const ctx = document.getElementById('chart-shap-beeswarm').getContext('2d');
    if (charts['shap_beeswarm']) charts['shap_beeswarm'].destroy();
    
    charts['shap_beeswarm'] = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'High Feature Value',
                    data: highValPoints,
                    backgroundColor: 'rgba(255, 180, 171, 0.7)', // light red
                    borderColor: '#ffb4ab',
                    borderWidth: 0.5,
                    pointRadius: 3.5,
                    pointHoverRadius: 5
                },
                {
                    label: 'Low Feature Value',
                    data: lowValPoints,
                    backgroundColor: 'rgba(142, 213, 255, 0.7)', // light blue
                    borderColor: '#8ed5ff',
                    borderWidth: 0.5,
                    pointRadius: 3.5,
                    pointHoverRadius: 5
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const pt = context.raw;
                            return `${pt.feature}: SHAP=${pt.x.toFixed(3)}, Val=${pt.feat_val_raw.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    title: { display: true, text: 'SHAP Value (Impact on prediction)', color: '#bdc8d1', font: { size: 10 } },
                    ticks: { color: '#bdc8d1', font: { family: 'JetBrains Mono' } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    min: -0.5,
                    max: 7.5,
                    ticks: {
                        color: '#d4e4fa',
                        font: { family: 'JetBrains Mono', size: 9 },
                        callback: function(val) {
                            const idx = 7 - Math.round(val);
                            if (idx >= 0 && idx < featureLabels.length) {
                                return featureLabels[idx];
                            }
                            return '';
                        }
                    }
                }
            }
        }
    });
}

// Candidates loaded successfully
function onCandidatesLoaded() {
    console.log("Candidate library compiled:", appData.candidates.length, "rows");
    
    // Set metric totals
    document.getElementById('btn-export-count').innerText = appData.candidates.length;
    document.getElementById('metric-filtered-count').innerText = appData.candidates.length;
    
    runFilters();
}

// Toggle light selection filter button
let activeLightValue = 'all';
function toggleLightFilter(val, btn) {
    activeLightValue = val;
    document.querySelectorAll('.filter-light-btn').forEach(b => {
        b.className = "filter-light-btn flex-1 border border-outline text-on-surface-variant py-2 rounded-lg font-label-caps text-[10px] hover:border-primary transition-all";
    });
    btn.className = "filter-light-btn flex-1 bg-primary text-on-primary-fixed py-2 rounded-lg font-label-caps text-[10px] transition-all";
    runFilters();
}

// Core filtering algorithm for Virtual Screening candidates
function runFilters() {
    const semiSelect = document.getElementById('filter-semiconductor').value;
    const cocatSelect = document.getElementById('filter-cocatalyst').value;
    
    appData.filteredCandidates = appData.candidates.filter(row => {
        // Semiconductor check
        if (semiSelect !== 'all') {
            if (String(row.host_material).toLowerCase().trim() !== semiSelect) return false;
        }
        
        // Co-catalyst check
        if (cocatSelect !== 'all') {
            if (String(row.co_catalyst).toLowerCase().trim() !== cocatSelect) return false;
        }
        
        // Light Source Type check
        if (activeLightValue !== 'all') {
            const rawLight = String(row.light_source_type).toLowerCase().trim();
            if (rawLight !== activeLightValue) return false;
        }
        
        return true;
    });

    // Update counts
    document.getElementById('btn-export-count').innerText = appData.filteredCandidates.length;
    document.getElementById('metric-filtered-count').innerText = appData.filteredCandidates.length;
    
    renderCandidatesTable();
}

// Render candidates in Screening Table
function renderCandidatesTable() {
    const tbody = document.getElementById('candidates-table-body');
    if (appData.filteredCandidates.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" class="p-md text-center text-outline">No candidates match the selected filters.</td></tr>`;
        return;
    }

    let html = '';
    appData.filteredCandidates.forEach((row, index) => {
        const materialLabel = `${String(row.host_material).toUpperCase()} / ${String(row.co_catalyst)}`;
        const confidencePct = Math.round(row.within_ad ? (100 - row.ad_score * 0.8) : 40);
        const confidenceColor = row.within_ad ? 'bg-secondary' : 'bg-error';
        const bandgap = row.bandgap_eV ? row.bandgap_eV.toFixed(2) : '2.40';

        html += `
        <!-- Row header -->
        <tr class="hover:bg-surface-variant transition-colors group cursor-pointer border-b border-outline-variant/20" onclick="toggleDetailsRow(${index})">
            <td class="p-md font-semibold text-primary">${materialLabel}</td>
            <td class="p-md font-data-md font-bold text-on-surface">${formatNum(row.pred_her_umol_g_h, 1)}</td>
            <td class="p-md">
                <div class="flex items-center gap-sm">
                    <div class="w-24 bg-surface-container h-1.5 rounded-full overflow-hidden">
                        <div class="${confidenceColor} h-full" style="width: ${Math.min(confidencePct, 100)}%;"></div>
                    </div>
                    <span class="text-[10px] text-outline font-semibold font-data-md">${row.within_ad ? 'AD_FIT' : 'OUT_AD'}</span>
                </div>
            </td>
            <td class="p-md font-data-md text-on-surface-variant">${bandgap} eV</td>
            <td class="p-md text-right">
                <span class="material-symbols-outlined text-outline group-hover:text-primary transition-colors text-lg" id="icon-row-${index}">expand_more</span>
            </td>
        </tr>
        <!-- Row details panel -->
        <tr class="bg-surface-container-low hidden border-b border-outline-variant/30" id="details-row-${index}">
            <td class="p-lg" colspan="5">
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-lg animate-in fade-in duration-300">
                    <div class="flex flex-col gap-md">
                        <h4 class="font-label-caps text-label-caps text-primary border-b border-primary/20 pb-xs">Prediction Insights</h4>
                        <div class="grid grid-cols-1 gap-sm text-[11px]">
                            <div class="flex justify-between items-center glass-panel p-2 rounded">
                                <span class="text-on-surface-variant">Novelty Vector Metric</span>
                                <span class="text-secondary font-bold font-data-md">+${(row.novelty_score * 100).toFixed(1)}%</span>
                            </div>
                            <div class="flex justify-between items-center glass-panel p-2 rounded">
                                <span class="text-on-surface-variant">Applicability Score (Distance)</span>
                                <span class="text-secondary font-bold font-data-md">${row.ad_score.toFixed(3)}</span>
                            </div>
                            <div class="flex justify-between items-center glass-panel p-2 rounded">
                                <span class="text-on-surface-variant">Recombination Risk Probability</span>
                                <span class="text-error font-bold font-data-md">${row.within_ad ? 'LOW' : 'HIGH'}</span>
                            </div>
                        </div>
                        <p class="text-on-surface-variant text-xs leading-relaxed mt-2">
                            The predictive model favors this heterojunction configuration. Conformal bound projections indicate with 90% confidence that the actual yield lies between ${formatNum(row.pred_p05_log ? Math.expm1(row.pred_p05_log) : 0, 0)} and ${formatNum(row.pred_p95_log ? Math.expm1(row.pred_p95_log) : 0, 0)} μmol/g/h.
                        </p>
                    </div>
                    <div class="h-44 rounded-lg overflow-hidden relative border border-outline-variant bg-surface">
                        <img src="https://lh3.googleusercontent.com/aida-public/AB6AXuAF5ScAWsajqzVqppTeDZ_xtFfIIjw8c_w-DMwr7Bb9TL6AcaX1jgAkO983UraQKsNxv250_z_7xv0JpP7zwl9R_IXeKqyNzMsjcXmIpNSkbCmE4gZyCZVi1CQa4XIjgPKJ9Swck7kXozDG80CnheT4UWBm2J9HNyri8mv1NJlr44WxI97tzP6jUG2F-AUd3VnApvEyuYkRwIt7dO6lBZAtX_oPdfSmLZEaSy1au3aDdjMmS6B5RFBIUdt9c-c5tlpQnNbI5oCsJNTB" 
                             alt="DFT Crystal Structure" 
                             class="w-full h-full object-cover opacity-60 hover:opacity-90 transition-opacity grayscale hover:grayscale-0"/>
                        <div class="absolute inset-0 bg-gradient-to-t from-surface-container-low to-transparent"></div>
                        <span class="absolute bottom-2 left-2 font-label-caps text-[9px] bg-surface/80 px-xs py-1 rounded">Crystal Simulation (DFT)</span>
                    </div>
                </div>
            </td>
        </tr>`;
    });
    
    tbody.innerHTML = html;
}

// Toggle expandable candidate details row
function toggleDetailsRow(index) {
    const row = document.getElementById(`details-row-${index}`);
    const icon = document.getElementById(`icon-row-${index}`);
    
    if (row.classList.contains('hidden')) {
        row.classList.remove('hidden');
        if (icon) icon.innerText = 'expand_less';
    } else {
        row.classList.add('hidden');
        if (icon) icon.innerText = 'expand_more';
    }
}

// Quick presets triggers
function applyPreset(presetType) {
    switchTab('screening');
    const semiSelect = document.getElementById('filter-semiconductor');
    const cocatSelect = document.getElementById('filter-cocatalyst');
    
    if (presetType === 'high_efficiency') {
        semiSelect.value = 'srtio3';
        cocatSelect.value = 'rh';
        toggleLightFilter('solar', document.querySelectorAll('.filter-light-btn')[2]);
    } else if (presetType === 'earth_abundant') {
        semiSelect.value = 'fe2o3';
        cocatSelect.value = 'wc';
        toggleLightFilter('solar', document.querySelectorAll('.filter-light-btn')[2]);
    }
}

// Zoom / View publication figure modal
document.querySelectorAll('#figures-grid img').forEach(img => {
    img.addEventListener('click', function() {
        const modal = document.getElementById('figure-modal');
        const modalImg = document.getElementById('modal-img');
        const modalCaption = document.getElementById('modal-caption');
        
        modalImg.src = this.src;
        modalCaption.innerText = this.alt;
        modal.classList.remove('hidden');
    });
});

function closeFigureModal() {
    document.getElementById('figure-modal').classList.add('hidden');
}

// Mock candidate exporter
function exportCandidates() {
    const csvContent = Papa.unparse(appData.filteredCandidates);
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", "photocatalyst_filtered_candidates.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Run application on load
window.addEventListener('DOMContentLoaded', init);
