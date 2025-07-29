# LateBench - Simplified Architecture

## Overview

LateBench is a mathematical reasoning error analysis framework focused on **late-occurring errors** in step-by-step solutions. This simplified version provides core functionality for error injection, critic evaluation, and manual curation through a clean, efficient codebase.

## 🎯 Core Components

### **Error Injection** (`src/core/error_injector.py`)
- **Purpose**: Inject natural mathematical errors into solution steps
- **Key Features**:
  - GPT-4 powered with refined prompts for natural-looking errors
  - Late error targeting (final 33% of solution steps)
  - Support for custom error suggestions
  - Rate limiting and error handling

### **Critic System** (`src/core/critic.py`)
- **Purpose**: Evaluate solutions for mathematical errors
- **Key Features**:
  - GPT-4o-mini based step-level error detection
  - Unified single/batch evaluation interface
  - Parallel processing with rate limiting
  - Detailed error explanations

### **Data Loader** (`src/core/data_loader.py`)
- **Purpose**: Load and manage mathematical reasoning datasets
- **Key Features**:
  - Unified interface for multiple dataset formats
  - Caching for efficient repeated access
  - Dataset statistics and filtering
  - Standardized LateBench format conversion

### **Metrics System** (`src/core/metrics.py`)
- **Purpose**: Evaluate critic performance with research-grade metrics
- **Key Features**:
  - DeltaBench-compatible step-level metrics
  - First-error cutoff logic
  - Precision, recall, F1-score calculation
  - Confusion matrix analysis

### **Utilities** (`src/utils/`)
- **Parallel Processing** (`parallel.py`): Thread-safe batch processing with rate limiting
- **Storage** (`storage.py`): Simple result storage and experiment organization

### **Dashboard** (`dashboard/simple_app.py`)
- **Purpose**: Interactive interface for manual error review and curation
- **Key Features**:
  - Dataset loading and navigation
  - Real-time error injection with custom suggestions
  - Manual decision tracking (Yes/Maybe/No)
  - Progress monitoring and export

## 🚀 Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Run Error Injection
```bash
# Basic error injection
python scripts/run_error_injection.py numinamath --max-examples 10

# With custom suggestion
python scripts/run_error_injection.py numinamath \
  --custom-suggestion "Make an invalid domain assumption" \
  --parallel --max-workers 5
```

### 3. Run Critic Evaluation
```bash
# Evaluate dataset
python scripts/run_critic_evaluation.py numinamath --compute-metrics

# Evaluate injection results
python scripts/run_critic_evaluation.py /path/to/injection/results \
  --model gpt-4o-mini --parallel
```

### 4. Launch Dashboard
```bash
# Start interactive dashboard
python scripts/run_dashboard.py --host localhost --port 8000

# Open http://localhost:8000 in browser
```

## 📁 Simplified File Structure

```
latebench/
├── src/core/                   # Core functionality
│   ├── error_injector.py       # Error injection system
│   ├── critic.py              # Critic evaluation system
│   ├── data_loader.py          # Data loading and management
│   └── metrics.py             # DeltaBench evaluation metrics
├── src/utils/                  # Utilities
│   ├── parallel.py            # Parallel processing
│   └── storage.py             # Result storage
├── dashboard/                  # Interactive dashboard
│   ├── simple_app.py          # Flask application
│   └── templates/             # HTML templates
├── scripts/                    # Main execution scripts
│   ├── run_error_injection.py # Error injection script
│   ├── run_critic_evaluation.py # Critic evaluation script
│   └── run_dashboard.py       # Dashboard launcher
└── data/                      # Data and results
    ├── datasets/              # LateBench format datasets
    └── results/               # Experiment results
```

## 🔄 Typical Workflow

### **Phase 1: Prompt Refinement**
1. **Test Current Prompts**:
   ```bash
   python scripts/run_error_injection.py numinamath --max-examples 20
   ```

2. **Manual Review**:
   ```bash
   python scripts/run_dashboard.py
   # Load results, review each error injection, mark quality
   ```

3. **Refine Prompts**:
   - Edit prompts in `src/core/error_injector.py`
   - Test improvements with new examples
   - Iterate until 80%+ approval rate

### **Phase 2: Dataset Creation**
1. **Large-Scale Injection**:
   ```bash
   python scripts/run_error_injection.py numinamath \
     --parallel --max-workers 10 --max-examples 500
   ```

2. **Manual Curation**:
   - Use dashboard to review all generated errors
   - Make Yes/Maybe/No decisions
   - Export approved examples

3. **Quality Control**:
   ```bash
   python scripts/run_critic_evaluation.py results_path --compute-metrics
   ```

## 🎛️ Configuration

### **Environment Variables**
- `OPENAI_API_KEY`: OpenAI API key (required)
- `REQUESTS_PER_MINUTE`: API rate limit (default: 60)

### **Model Selection**
- **Error Injection**: `gpt-4-turbo-preview` (default)
- **Critic Evaluation**: `gpt-4o-mini` (default)

### **Parallel Processing**
- **Error Injection**: 8-10 workers recommended
- **Critic Evaluation**: 6-8 workers recommended
- **Rate Limiting**: Automatic across all threads

## 📊 Expected Performance

### **Error Injection**
- **Success Rate**: 95%+ for solutions with 4+ steps
- **Processing Speed**: ~3-5 seconds per problem
- **Quality**: 80%+ approval rate with refined prompts

### **Critic Evaluation**
- **Processing Speed**: ~2-3 seconds per problem
- **Detection Rate**: 40-50% recall, 10-20% precision (typical)
- **Parallel Speedup**: 6-8x with optimal worker count

## 🛠️ Development

### **Adding New Error Types**
1. Update prompts in `ErrorInjector._create_system_prompt()`
2. Test with dashboard manual review
3. Validate with critic evaluation

### **Supporting New Datasets**
1. Add processor to convert to LateBench format
2. Save as `latebench_{name}_raw.json` in `data/datasets/`
3. Test with data loader

### **Customizing Metrics**
1. Modify `DeltaBenchEvaluator` in `src/core/metrics.py`
2. Update evaluation scripts to use new metrics
3. Test with known ground truth data

## 🎯 Success Criteria

- **Error Quality**: 80%+ manual approval rate
- **System Reliability**: 95%+ success rate for all operations
- **Processing Efficiency**: <5 seconds per error injection
- **Workflow Simplicity**: Complete prompt refinement cycle in <1 day

This simplified architecture maintains all essential functionality while providing a clean, efficient foundation for mathematical reasoning error analysis research.