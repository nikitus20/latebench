// LateBench Dashboard JavaScript

// Global variables
let currentExampleId = null;
let allExamples = [];
let filteredExamples = [];

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    // Load initial data if available
    if (typeof window.currentExampleId !== 'undefined') {
        currentExampleId = window.currentExampleId;
    }
    
    if (typeof window.allExamples !== 'undefined') {
        allExamples = window.allExamples;
        filteredExamples = [...allExamples];
    }
    
    // Set up keyboard shortcuts
    setupKeyboardShortcuts();
    
    // Initialize MathJax if available
    initializeMathJax();
});

// Keyboard shortcuts
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(event) {
        // Don't trigger shortcuts if user is typing in an input
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA' || event.target.tagName === 'SELECT') {
            return;
        }
        
        switch(event.key) {
            case 'ArrowLeft':
                event.preventDefault();
                previousExample();
                break;
            case 'ArrowRight':
                event.preventDefault();
                nextExample();
                break;
            case 'r':
            case 'R':
                if (event.ctrlKey || event.metaKey) {
                    event.preventDefault();
                    runCritic();
                }
                break;
            case 'e':
            case 'E':
                if (event.ctrlKey || event.metaKey) {
                    event.preventDefault();
                    exportExample();
                }
                break;
        }
    });
}

// MathJax initialization
function initializeMathJax() {
    if (typeof MathJax !== 'undefined') {
        // Re-render math when content changes
        MathJax.typesetPromise().then(() => {
            console.log('MathJax rendering complete');
        }).catch((err) => {
            console.warn('MathJax rendering error:', err);
        });
    }
}

// Navigation functions
function selectExample(exampleId) {
    window.location.href = `/example/${exampleId}`;
}

function previousExample() {
    if (!currentExampleId || filteredExamples.length === 0) return;
    
    const currentIndex = filteredExamples.findIndex(ex => ex.id === currentExampleId);
    if (currentIndex > 0) {
        selectExample(filteredExamples[currentIndex - 1].id);
    } else {
        // Wrap to last example
        selectExample(filteredExamples[filteredExamples.length - 1].id);
    }
}

function nextExample() {
    if (!currentExampleId || filteredExamples.length === 0) return;
    
    const currentIndex = filteredExamples.findIndex(ex => ex.id === currentExampleId);
    if (currentIndex < filteredExamples.length - 1) {
        selectExample(filteredExamples[currentIndex + 1].id);
    } else {
        // Wrap to first example
        selectExample(filteredExamples[0].id);
    }
}

function jumpToExample(exampleId) {
    if (exampleId) {
        selectExample(exampleId);
    }
}

// Critic evaluation
async function runCritic() {
    if (!currentExampleId) {
        showNotification('No example selected', 'error');
        return;
    }
    
    const btn = document.getElementById('critic-btn');
    const btnText = document.getElementById('critic-btn-text');
    const spinner = document.getElementById('critic-spinner');
    
    if (!btn || !btnText || !spinner) {
        console.warn('Critic button elements not found');
        return;
    }
    
    // Show loading state
    btn.disabled = true;
    btnText.textContent = 'Running...';
    spinner.style.display = 'inline-block';
    
    try {
        showNotification('Running critic evaluation...', 'info');
        
        const response = await fetch(`/api/run_critic/${currentExampleId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const result = await response.json();
        
        if (result.success) {
            showNotification('Critic evaluation complete!', 'success');
            // Reload page to show results
            setTimeout(() => {
                window.location.reload();
            }, 1000);
        } else {
            showNotification('Error running critic: ' + result.error, 'error');
        }
    } catch (error) {
        showNotification('Error running critic: ' + error.message, 'error');
        console.error('Critic evaluation error:', error);
    } finally {
        // Reset button state
        btn.disabled = false;
        btnText.textContent = 'Run Critic';
        spinner.style.display = 'none';
    }
}

// Export function
function exportExample() {
    if (!currentExampleId) {
        showNotification('No example selected', 'error');
        return;
    }
    
    const exportUrl = `/export/${currentExampleId}`;
    
    // Open in new tab
    window.open(exportUrl, '_blank');
    
    // Also trigger download
    const link = document.createElement('a');
    link.href = exportUrl;
    link.download = `example_${currentExampleId}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    showNotification('Example exported', 'success');
}

// Dataset switching functions
async function switchDataset() {
    const selector = document.getElementById('dataset-selector');
    if (!selector) return;
    
    const selectedValue = selector.value;
    if (!selectedValue) return;
    
    const [datasetName, problemType] = selectedValue.split('|');
    
    try {
        showNotification('Switching dataset...', 'info');
        
        const response = await fetch('/api/switch_dataset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                dataset_name: datasetName,
                problem_type: problemType
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Reload the page to show new dataset
            showNotification(`Switched to ${datasetName} (${problemType})`, 'success');
            window.location.reload();
        } else {
            showNotification(`Failed to switch dataset: ${data.error}`, 'error');
        }
        
    } catch (error) {
        showNotification(`Error switching dataset: ${error.message}`, 'error');
    }
}

// Filtering functions
async function applyFilters() {
    const errorType = document.getElementById('error-type-filter')?.value;
    const minSteps = document.getElementById('min-steps')?.value;
    const maxSteps = document.getElementById('max-steps')?.value;
    const hasCritic = document.getElementById('has-critic-filter')?.checked;
    
    const params = new URLSearchParams();
    if (errorType && errorType !== 'all') params.append('error_type', errorType);
    if (minSteps) params.append('min_steps', minSteps);
    if (maxSteps) params.append('max_steps', maxSteps);
    if (hasCritic) params.append('has_critic', 'true');
    
    try {
        showNotification('Applying filters...', 'info');
        
        const response = await fetch(`/api/filter?${params}`);
        const data = await response.json();
        
        filteredExamples = data.examples;
        updateProblemList(data.examples);
        
        const countElement = document.getElementById('problem-count');
        if (countElement) {
            countElement.textContent = `(${data.count})`;
        }
        
        showNotification(`Found ${data.count} matching examples`, 'success');
    } catch (error) {
        showNotification('Error applying filters: ' + error.message, 'error');
        console.error('Filter error:', error);
    }
}

function clearFilters() {
    const errorTypeFilter = document.getElementById('error-type-filter');
    const minStepsFilter = document.getElementById('min-steps');
    const maxStepsFilter = document.getElementById('max-steps');
    const hasCriticFilter = document.getElementById('has-critic-filter');
    
    if (errorTypeFilter) errorTypeFilter.value = 'all';
    if (minStepsFilter) minStepsFilter.value = '';
    if (maxStepsFilter) maxStepsFilter.value = '';
    if (hasCriticFilter) hasCriticFilter.checked = false;
    
    applyFilters();
}

function updateProblemList(examples) {
    const container = document.getElementById('problems-container');
    if (!container) return;
    
    container.innerHTML = '';
    
    examples.forEach(example => {
        const card = document.createElement('div');
        card.className = 'problem-card' + (example.id === currentExampleId ? ' selected' : '');
        card.setAttribute('data-id', example.id);
        card.onclick = () => selectExample(example.id);
        
        const errorType = example.error_info.type.replace(/_/g, ' ');
        const errorTypeCapitalized = errorType.split(' ').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
        
        card.innerHTML = `
            <div class="problem-title">${escapeHtml(example.title)}</div>
            <div class="problem-meta">
                <span class="error-type">${errorTypeCapitalized}</span>
                <span class="step-info">${example.original_solution.num_steps} steps</span>
            </div>
            <div class="problem-status">
                ${example.critic_result ? '<span class="critic-badge">✓ Analyzed</span>' : ''}
                <span class="error-step">Error: Step ${example.error_info.step}</span>
            </div>
        `;
        
        container.appendChild(card);
    });
}

// Utility functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existing = document.querySelectorAll('.notification');
    existing.forEach(n => n.remove());
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Style the notification
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        border-radius: 4px;
        color: white;
        font-weight: 500;
        z-index: 1000;
        max-width: 400px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        animation: slideIn 0.3s ease-out;
    `;
    
    // Set background color based on type
    switch (type) {
        case 'success':
            notification.style.backgroundColor = '#28a745';
            break;
        case 'error':
            notification.style.backgroundColor = '#dc3545';
            break;
        case 'warning':
            notification.style.backgroundColor = '#ffc107';
            notification.style.color = '#212529';
            break;
        default:
            notification.style.backgroundColor = '#007bff';
    }
    
    // Add animation styles if not already present
    if (!document.getElementById('notification-styles')) {
        const styles = document.createElement('style');
        styles.id = 'notification-styles';
        styles.textContent = `
            @keyframes slideIn {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
            
            @keyframes slideOut {
                from {
                    transform: translateX(0);
                    opacity: 1;
                }
                to {
                    transform: translateX(100%);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(styles);
    }
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.style.animation = 'slideOut 0.3s ease-in forwards';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.remove();
                }
            }, 300);
        }
    }, 5000);
    
    // Allow manual dismissal
    notification.addEventListener('click', () => {
        if (notification.parentNode) {
            notification.style.animation = 'slideOut 0.3s ease-in forwards';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.remove();
                }
            }, 300);
        }
    });
}

// Math rendering helper
function rerenderMath() {
    if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
        MathJax.typesetPromise().then(() => {
            console.log('Math re-rendered');
        }).catch((err) => {
            console.warn('Math rendering error:', err);
        });
    }
}

// API helper functions
async function fetchAPI(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

// Make functions globally available
window.selectExample = selectExample;
window.previousExample = previousExample;
window.nextExample = nextExample;
window.jumpToExample = jumpToExample;
window.runCritic = runCritic;
window.exportExample = exportExample;
window.applyFilters = applyFilters;
window.clearFilters = clearFilters;