# LateBench: Mathematical Reasoning Error Analysis Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Dashboard](https://img.shields.io/badge/Dashboard-Live-green.svg)](http://localhost:8000)

LateBench is a comprehensive framework for analyzing mathematical reasoning errors in large language models (LLMs). It provides tools for error detection, systematic error injection, and evaluation of AI models' ability to identify mathematical mistakes in step-by-step reasoning.

## 🎯 Project Overview

LateBench combines natural error analysis from human-annotated datasets with systematic error injection techniques to create a robust testing environment for mathematical reasoning capabilities. The framework enables researchers to:

- **Analyze Natural Errors**: Study real human-verified mathematical errors from existing datasets
- **Inject Systematic Errors**: Create controlled error scenarios using adversarial techniques  
- **Evaluate Critic Models**: Test LLMs' ability to detect and locate mathematical errors
- **Manual Annotation**: Interactive dashboard for curating high-quality error datasets
- **Comprehensive Analysis**: Step-by-step reasoning evaluation with detailed metrics

## 🚀 Current Project Status

### ✅ Core Features Implemented

#### 📊 **Multi-Dataset Support**
- **PRM800K Integration**: Complete processing of human-annotated mathematical reasoning dataset
- **NuminaMath Support**: High-quality mathematical competition problems with detailed solutions
- **Unified Data Schema**: Consistent `LateBenchExample` format across all datasets

#### 🎯 **Error Analysis Pipeline**
- **Human Annotation Parsing**: Extracts error step information from PRM800K human ratings
- **Adversarial Error Injection**: GPT-4 powered systematic error introduction with custom prompts
- **Step-Level Error Detection**: Granular analysis of where reasoning breaks down
- **Error Type Classification**: Detailed categorization of mathematical error patterns

#### 🔍 **Interactive Dashboard**
- **Problem Navigation**: Browse through mathematical problems with filtering and quick-jump
- **Manual Error Injection**: Interactive interface for custom error suggestion and injection
- **Real-Time Progress Tracking**: Visual feedback for error injection and critic evaluation
- **Per-Problem State Management**: Intelligent button states preventing race conditions
- **Decision Workflow**: Yes/Maybe/No decisions for curating final datasets

#### 🤖 **LLM Critic Evaluation**
- **Independent Error Detection**: GPT-4o-mini powered critic for mathematical reasoning validation
- **Step-by-Step Analysis**: Detailed explanations for each detected error
- **Ground Truth Comparison**: Performance metrics against human annotations
- **Clean Data Evaluation**: Bias-free assessment without reasoning type markers

#### 📈 **Complete Workflow Implementation**
- **Data Processing**: PRM800K → LateBench unified format → Dashboard ready
- **Manual Injection**: Problem → Custom suggestion → GPT-4 injection → Review → Decision
- **Critic Analysis**: Solution → Independent evaluation → Error detection → Comparison
- **Dataset Creation**: Curated examples → Final LateBench dataset with quality decisions

### 🗂️ **Clean Project Structure**

```
latebench/
├── src/                          # Core library code
│   ├── data_processing/          # Dataset processors and unified schema
│   │   ├── unified_schema.py     # LateBenchExample data format
│   │   ├── prm800k_processor.py  # PRM800K human annotation processing
│   │   └── numinamath_processor.py # NuminaMath competition problems
│   ├── error_injector.py         # Adversarial error injection system
│   ├── critic.py                 # LLM critic evaluation framework
│   ├── dataset_manager.py        # Unified dataset loading and management
│   └── error_types.py            # Mathematical error classification
├── dashboard/                    # Interactive web interface
│   ├── app.py                    # Flask application with API endpoints
│   ├── utils.py                  # Data integration and processing utilities
│   ├── static/                   # CSS, JavaScript, styling
│   └── templates/                # HTML templates with step visualization
├── tests/                        # Comprehensive test suite
│   ├── test_core_system.py       # Core system validation
│   ├── test_api_endpoints.py     # Dashboard API testing
│   ├── test_data_integrity.py    # Data processing verification
│   └── test_error_injection.py   # Complete workflow testing
├── data/                         # Clean data structure
│   ├── datasets/                 # Core datasets only
│   │   ├── latebench_prm800k_raw.json      # Full PRM800K with annotations
│   │   └── latebench_numinamath_raw.json   # NuminaMath competition problems
│   ├── manual_injection_data.json          # Manual annotation decisions
│   ├── dashboard_critic_results.json       # Critic evaluation results
│   └── sources/                  # Original dataset sources
├── logs/                         # Application and processing logs
├── run_dashboard.py              # Dashboard entry point
└── requirements.txt              # Python dependencies
```

### 🔧 **Available Datasets**

- **`prm800k`**: 800K+ mathematical reasoning examples with human step annotations
- **`numinamath`**: High-quality competition mathematics problems

Both datasets are available in unified LateBench format with proper error step detection.

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- OpenAI API key (for error injection and critic evaluation)

### Quick Setup

1. **Clone and setup environment**
   ```bash
   git clone https://github.com/your-username/latebench.git
   cd latebench
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure OpenAI API** (for error injection and critic features)
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Download datasets**
   ```bash
   python download_prm800k.py
   ```

4. **Launch dashboard**
   ```bash
   python run_dashboard.py
   ```

   Open http://localhost:8000 to access the interactive interface.

## 🚀 Usage Examples

### Dashboard Interface

The interactive dashboard provides a complete workflow for mathematical reasoning analysis:

1. **Browse Problems**: Navigate through 100+ mathematical problems with human annotations
2. **View Error Steps**: See step-by-step solutions with error highlighting based on human ratings
3. **Manual Error Injection**: 
   - Add custom error suggestions
   - Generate adversarial examples with GPT-4
   - Track progress with visual feedback
4. **Critic Evaluation**: Run independent LLM analysis to detect errors
5. **Make Decisions**: Curate high-quality examples with Yes/Maybe/No decisions

### Programmatic Usage

```python
from src.dataset_manager import LateBenchDatasetManager
from src.critic import LLMCritic
from src.error_injector import AdversarialErrorInjector

# Load datasets
manager = LateBenchDatasetManager()
manager.load_dataset('prm800k', 'all')
examples = manager.get_current_examples()

# Analyze with LLM critic
critic = LLMCritic()
for example in examples[:5]:
    result = critic.evaluate_solution(example)
    print(f"Problem: {example.id}")
    print(f"Human Error Step: {example.get_first_error_step()}")
    print(f"Critic Detected: {result.has_errors}")
    print(f"Critic Error Steps: {result.error_steps}")

# Inject systematic errors
injector = AdversarialErrorInjector()
modified_example = injector.inject_error(
    example=examples[0],
    error_suggestion="Make an algebraic manipulation error in step 5"
)
```

## 🧪 Testing

LateBench includes a comprehensive test suite:

```bash
# Run all tests
python run_tests.py

# Individual test categories
python -m pytest tests/test_core_system.py      # Core functionality
python -m pytest tests/test_api_endpoints.py    # Dashboard API
python -m pytest tests/test_data_integrity.py   # Data processing
python -m pytest tests/test_error_injection.py  # Complete workflow
```

## 📊 Key Metrics & Features

### Error Detection Performance
- **Human Annotation Integration**: Proper parsing of PRM800K step ratings (-1: error, 0: questionable, 1: correct)
- **Critic Evaluation**: Independent LLM assessment with detailed error explanations
- **Step-Level Analysis**: Granular understanding of reasoning breakdown points

### Dashboard Features
- **100+ Examples**: Loaded from core PRM800K dataset with human annotations
- **Real-Time Feedback**: Progress bars and status updates during processing
- **Intelligent UI**: Per-problem button states and race condition prevention
- **Decision Tracking**: Persistent storage of annotation decisions and timing

### Data Quality
- **Clean Architecture**: Simplified to core datasets only (no intermediate files)
- **Unified Format**: Consistent `LateBenchExample` schema across all data
- **Proper Error Steps**: Human annotation error steps correctly extracted and displayed

## 🤝 Development

### Current Status: Production Ready ✅

The framework is fully functional with:
- ✅ Complete error injection and critic evaluation workflow
- ✅ Interactive dashboard with manual annotation capabilities  
- ✅ Proper human annotation parsing and error step detection
- ✅ Clean codebase with comprehensive test coverage
- ✅ Ready for dataset curation and research experiments

### Next Steps for Users

1. **Data Filtering**: Implement custom filtering logic for specific research needs
2. **Dataset Expansion**: Add additional mathematical reasoning datasets
3. **Advanced Analysis**: Develop domain-specific error pattern recognition
4. **Research Applications**: Use framework for LLM evaluation and improvement

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Documentation**: This README provides comprehensive usage information
- **Dashboard**: Interactive interface at http://localhost:8000 when running

---

**LateBench** - A comprehensive framework for understanding mathematical reasoning errors in AI systems. 🔢✨