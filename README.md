# LateBench: Mathematical Reasoning Error Analysis Framework

LateBench is a comprehensive benchmarking framework for analyzing mathematical reasoning errors in large language models (LLMs). It provides tools for error injection, detection, and analysis across multiple mathematical datasets, with a focus on understanding where and why AI models make mistakes in step-by-step reasoning.

## 🎯 Overview

LateBench combines natural error analysis from human-annotated datasets (like PRM800K) with systematic error injection techniques to create a robust testing environment for mathematical reasoning capabilities. The framework enables researchers to:

- **Analyze Natural Errors**: Study real human-verified mathematical errors from existing datasets
- **Inject Systematic Errors**: Create controlled error scenarios for testing model robustness  
- **Evaluate Critic Models**: Test LLMs' ability to detect and locate mathematical errors
- **Visualize Results**: Interactive dashboard for exploring errors and model performance

## 🚀 Key Features

### 📊 Multi-Dataset Support
- **PRM800K Integration**: 500+ problems with human-verified step-by-step error annotations
- **NuminaMath Support**: High-quality mathematical problems with detailed solutions
- **Unified Data Format**: Consistent schema across all datasets for easy analysis

### 🎯 Error Analysis Capabilities
- **Natural Error Detection**: Human-annotated errors from mathematical reasoning steps
- **Systematic Error Injection**: Controlled introduction of mathematical mistakes
- **Step-Level Analysis**: Granular understanding of where reasoning breaks down
- **Error Type Classification**: Categorization of different mathematical error patterns

### 🔍 Interactive Dashboard
- **Visual Error Exploration**: Color-coded step importance and error highlighting
- **Problem Navigation**: Browse through mathematical problems with filtering options
- **Real-Time Analysis**: Interactive exploration of model predictions vs. ground truth
- **Performance Metrics**: Comprehensive statistics on error detection accuracy

### 🤖 LLM Integration
- **Critic Model Evaluation**: Test language models' ability to identify mathematical errors
- **Multiple Model Support**: Compatible with various LLM architectures
- **Configurable Prompting**: Customizable prompts for different evaluation scenarios

## 📁 Project Structure

```
latebench/
├── src/                          # Core library code
│   ├── data_processing/          # Dataset processors and unified schema
│   │   ├── unified_schema.py     # Common data format definitions
│   │   ├── prm800k_processor.py  # PRM800K dataset processor
│   │   └── numinamath_processor.py # NuminaMath processor
│   ├── error_injector.py         # Error injection system
│   ├── critic.py                 # LLM critic evaluation
│   ├── dataset_manager.py        # Dataset loading and management
│   └── error_types.py            # Error classification system
├── dashboard/                    # Web interface
│   ├── app.py                    # Flask application
│   ├── utils.py                  # Dashboard utilities
│   ├── static/                   # CSS, JavaScript, images
│   └── templates/                # HTML templates
├── scripts/                      # Experiment and utility scripts
├── data/                         # Data directory (excluded from git)
│   ├── datasets/                 # Processed datasets
│   └── experiments/              # Experiment results
├── logs/                         # Application logs (excluded from git)
├── run_dashboard.py              # Dashboard entry point
└── requirements.txt              # Python dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/nikitus20/latebench.git
   cd latebench
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and process datasets**
   ```bash
   # Download PRM800K dataset
   python download_prm800k.py
   
   # Process datasets (this will create unified format files in data/datasets/)
   python -c "
   from src.dataset_manager import LateBenchDatasetManager
   manager = LateBenchDatasetManager()
   manager.process_all_datasets()
   "
   ```

## 🚀 Quick Start

### 1. Launch the Dashboard

```bash
python run_dashboard.py
```

Open your browser to http://localhost:8000 to explore the interactive interface.

### 2. Analyze Mathematical Errors

The dashboard provides:
- **Problem Browser**: Navigate through 500+ mathematical problems
- **Step Visualization**: See step-by-step solutions with error highlighting
- **Error Detection**: Human-verified errors marked with importance levels:
  - 🔴 **High**: Error steps requiring attention
  - 🟡 **Medium**: Correct but important steps  
  - 🟢 **Low**: Questionable or less critical steps

### 3. Run Error Analysis

```python
from src.dataset_manager import LateBenchDatasetManager
from src.critic import LLMCritic

# Load data
manager = LateBenchDatasetManager()
examples = manager.get_examples(dataset_name="prm800k", limit=10)

# Analyze with LLM critic
critic = LLMCritic()
for example in examples:
    result = critic.evaluate_solution(example)
    print(f"Problem: {example.id}")
    print(f"Detected Error: {result.has_error}")
```

## 📊 Dataset Information

### PRM800K Integration
- **Source**: Human-annotated mathematical reasoning dataset
- **Coverage**: 500+ problems with step-by-step error analysis
- **Error Types**: Human-verified mistakes in mathematical reasoning
- **Annotations**: Step-level ratings (-1: error, 0: questionable, 1: correct)

### NuminaMath Support  
- **Source**: High-quality mathematical competition problems
- **Focus**: Complex multi-step reasoning scenarios
- **Integration**: Unified format compatible with error injection system

## 🔧 Configuration

### Environment Variables

The dashboard and core system support configuration via environment variables:

```bash
# Dashboard configuration
export PORT=8080                    # Dashboard port (default: 8000)
export HOST=127.0.0.1               # Host address (default: 127.0.0.1)  
export DEBUG=false                  # Debug mode (default: true)

# Data paths
export LATEBENCH_DATA_DIR=./data    # Data directory
export LATEBENCH_LOG_DIR=./logs     # Log directory
```

### Custom Dataset Integration

To add new datasets:

1. **Create a processor** in `src/data_processing/`
2. **Implement the unified schema** using `LateBenchExample` format
3. **Register with the dataset manager** in `src/dataset_manager.py`

Example processor structure:
```python
from .unified_schema import LateBenchExample, LateBenchStep

class CustomProcessor:
    def process_example(self, raw_data):
        # Convert raw data to LateBenchExample format
        steps = [LateBenchStep(...) for step in raw_data['steps']]
        return LateBenchExample(steps=steps, ...)
```

## 🧪 Experiments and Scripts

### Available Scripts

- **`scripts/check_progress.py`**: Monitor dataset processing progress
- **`scripts/create_large_dataset.py`**: Generate large-scale evaluation datasets
- **`scripts/run_large_scale_injection.py`**: Execute batch error injection experiments

### Running Experiments

```bash
# Large-scale error injection experiment
python scripts/run_large_scale_injection.py --dataset prm800k --num_examples 1000

# Check processing progress
python scripts/check_progress.py --dataset all
```

## 📈 Performance and Metrics

LateBench provides comprehensive metrics for mathematical reasoning evaluation:

### Error Detection Metrics
- **Precision**: Accuracy of error identification
- **Recall**: Coverage of actual errors
- **F1-Score**: Balanced error detection performance
- **Step-Level Accuracy**: Granular reasoning step evaluation

### Analysis Features
- **Error Type Distribution**: Classification of mathematical mistake patterns
- **Difficulty Analysis**: Performance correlation with problem complexity
- **Step Importance**: Understanding critical reasoning points
- **Model Comparison**: Side-by-side LLM performance analysis

## 🤝 Contributing

We welcome contributions to LateBench! Please see our contribution guidelines:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/new-analysis`)
3. **Make your changes** with proper tests and documentation
4. **Submit a pull request** with a clear description

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black src/ dashboard/ scripts/
isort src/ dashboard/ scripts/
```

## 📝 Logging and Debugging

LateBench includes comprehensive logging:

- **Dashboard Logs**: `logs/dashboard_YYYYMMDD_HHMMSS.log`
- **Processing Logs**: Detailed dataset processing information
- **Error Tracking**: Automatic error logging and debugging information

Log levels can be configured via environment variables:
```bash
export LATEBENCH_LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
```

## 📚 Citation

If you use LateBench in your research, please cite:

```bibtex
@software{latebench2024,
  title={LateBench: Mathematical Reasoning Error Analysis Framework},
  author={LateBench Contributors},
  year={2024},
  url={https://github.com/nikitus20/latebench},
  note={A comprehensive framework for analyzing mathematical reasoning errors in large language models}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support and Issues

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive guides and API documentation
- **Community**: Join discussions about mathematical reasoning evaluation

## 🔮 Roadmap

### Upcoming Features
- [ ] Additional dataset integrations (GSM8K, MATH, etc.)
- [ ] Advanced error pattern analysis
- [ ] Multi-modal mathematical reasoning support
- [ ] Automated error classification
- [ ] Enhanced visualization tools
- [ ] API endpoints for external integration

### Long-term Goals
- Standardized mathematical reasoning evaluation protocol
- Integration with popular ML frameworks
- Large-scale benchmarking infrastructure
- Community-driven error taxonomy development

---

**LateBench** - Understanding where mathematical reasoning goes wrong, one step at a time. 🔢✨