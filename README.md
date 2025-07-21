# LateBench: Benchmark for Late Reasoning Errors in Math Solutions

LateBench is a tool for creating adversarial examples by injecting logical errors late in mathematical reasoning chains. It uses the NuminaMath dataset and GPT-4 to generate challenging test cases for reasoning critics and process reward models.

## Overview

This project creates a dataset of mathematical problems with intentionally injected late-appearing logical errors to test the capabilities of process reward models and reasoning critics. The errors are designed to:

- Appear plausible at first glance
- Use valid mathematical notation
- Lead to incorrect final answers through flawed reasoning
- Require careful analysis to detect
- Occur in the last 25% of solution steps

## Features

- **12 Types of Logical Errors**: Including invalid generalizations, theorem misapplication, circular reasoning, etc.
- **Automated Error Injection**: Uses GPT-4 to intelligently inject context-appropriate errors
- **Quality Control**: Validates error placement, final answer differences, and logical consistency  
- **Comprehensive Analysis**: Generates statistics, visualizations, and HTML reports
- **Large-Scale Dataset**: Built on 859K+ examples from NuminaMath-CoT

## Quick Start

### 1. Setup Environment

```bash
# Clone and setup
cd latebench
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 2. Download Dataset

```bash
python download_data.py
```

This downloads the NuminaMath dataset (~859K examples) and filters for problems with 8+ solution steps.

### 3. Run Quick Test

```bash
python test_implementation.py
```

### 4. Run Small Experiment

```bash
python run_experiment.py --experiment small --num_examples 5
```

### 5. Launch Interactive Dashboard

```bash
python start_dashboard.py
```

Open http://localhost:8000 in your browser for the interactive dashboard with:
- **Four-panel layout**: Problem statement, original solution, modified solution, critic analysis
- **Problem browser**: Filter by error type, step count, critic analysis status
- **One-click critic evaluation**: Run GPT-4o-mini to detect mathematical errors
- **Performance analysis**: Compare critic predictions vs ground truth errors
- **Export functionality**: Save examples as JSON for further analysis

### 6. Alternative: Static Analysis

For non-interactive analysis, check:
- `data/sample_injection.md` - Example of injected error
- `data/experiment_stats.png` - Statistical plots
- `notebooks/example_generation.ipynb` - Interactive Jupyter exploration

## Architecture

### Core Components

- **`data_loader.py`** - NuminaMath dataset loading and analysis
- **`error_types.py`** - Taxonomy of 12 logical error types
- **`error_injector.py`** - GPT-4 powered error injection system
- **`critic.py`** - GPT-4o-mini critic for error detection
- **`dashboard.py`** - Interactive web interface for manual inspection
- **`visualization.py`** - Analysis and reporting tools

### Error Types

1. **Invalid Generalization** - Extending specific cases without justification
2. **Theorem Misapplication** - Using theorems outside valid conditions
3. **Incomplete Case Analysis** - Missing important cases or boundary conditions  
4. **Circular Reasoning** - Using conclusions to prove themselves
5. **False Equivalence** - Treating non-equivalent statements as equivalent
6. **Domain Restriction Violation** - Operations outside valid domains
7. **Quantifier Confusion** - Mixing universal/existential quantifiers
8. **Invalid Substitution** - Substituting under invalid conditions
9. **False Contrapositive** - Incorrectly applying logical relationships
10. **Invalid Inverse Operation** - Applying inverses without considering restrictions
11. **Composition Error** - Incorrectly composing functions/operations
12. **Boundary Condition Error** - Mishandling edge cases

## Usage Examples

### Generate Single Example

```python
from src.data_loader import NuminaMathDataLoader
from src.error_injector import AdversarialErrorInjector

# Load data
loader = NuminaMathDataLoader()
problems = loader.get_sample_examples(n=1, min_steps=8)

# Inject error
injector = AdversarialErrorInjector()
result = injector.inject_error(problems[0], error_type_preference="circular_reasoning")

if result.success:
    print(f"Error injected at step {result.error_analysis['selected_error_step']}")
```

### Batch Processing

```python
# Process multiple examples
error_distribution = {
    "invalid_generalization": 0.3,
    "theorem_misapplication": 0.3, 
    "circular_reasoning": 0.4
}

results = injector.batch_inject_errors(
    problems[:10], 
    error_distribution=error_distribution
)
```

### Critic Evaluation

```python
from src.critic import LLMCritic, evaluate_single_example

# Initialize critic
critic = LLMCritic(model="gpt-4o-mini")

# Evaluate single example
result = evaluate_single_example(problem_dict)
print(f"Errors found: {result.has_errors}")
print(f"Error steps: {result.error_steps}")

# Or use the interactive dashboard
# python start_dashboard.py
```

### Analysis

```python
from src.visualization import VISUALIZER

# Generate quality report
metrics = VISUALIZER.create_quality_metrics_report(results)
print(f"Success rate: {metrics['overall_metrics']['success_rate']:.1%}")

# Create visualizations
VISUALIZER.create_batch_statistics_plot(results, save_path='stats.png')
VISUALIZER.save_html_report(results, 'report.html')
```

## Configuration

### Environment Variables (.env)

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview  # Optional
REQUESTS_PER_MINUTE=60           # Optional, rate limiting
```

### Error Distribution

Customize error types in experiments:

```python
error_distribution = {
    "invalid_generalization": 0.25,
    "theorem_misapplication": 0.25,
    "circular_reasoning": 0.25,
    "domain_restriction_violation": 0.25
}
```

## File Structure

```
latebench/
├── src/                    # Core implementation
│   ├── data_loader.py     # Dataset handling
│   ├── error_types.py     # Error taxonomy
│   ├── error_injector.py  # Error injection engine
│   └── visualization.py   # Analysis tools
├── notebooks/             # Jupyter notebooks
├── data/                  # Datasets and results
├── configs/               # Configuration files  
├── tests/                 # Test files
├── download_data.py       # Dataset download script
├── run_experiment.py      # Experiment runner
├── test_implementation.py # Validation tests
└── requirements.txt       # Dependencies
```

## Advanced Usage

### Large-Scale Dataset Creation

```bash
# Generate 1000 examples with full error distribution
python run_experiment.py --experiment full --num_examples 1000
```

### Custom Error Types

Add new error types to `error_types.py`:

```python
"custom_error": ErrorType(
    name="custom_error",
    description="Your error description",
    example="Example of the error",
    mathematical_context="When this error typically occurs",
    detection_difficulty="High - why it's hard to detect",
    common_subjects=["algebra", "calculus"]
)
```

### Integration with Evaluation

```python
# Use generated examples to test reasoning critics
adversarial_examples = injector.load_results("data/full_experiment_results.json")

for result in adversarial_examples:
    if result.success:
        # Test your model on result.modified_solution
        prediction = your_model.detect_error(result.modified_solution)
        # Evaluate against known error location
```

## Quality Metrics

The system tracks several quality indicators:

- **Success Rate**: Percentage of successful error injections
- **Late Position Compliance**: Errors placed in last 25% of solution
- **Error Type Diversity**: Distribution across different error categories
- **Detection Difficulty**: Proportion of high-difficulty errors
- **Answer Differentiation**: All modified solutions have different final answers

## Research Applications

LateBench is designed for:

1. **Process Reward Model Training**: Generate training data with step-level error labels
2. **Reasoning Critic Evaluation**: Test model ability to detect subtle logical errors
3. **Mathematical Reasoning Research**: Study failure modes in multi-step reasoning
4. **Adversarial Robustness**: Test model performance on challenging edge cases

## Citation

If you use LateBench in your research, please cite:

```bibtex
@software{latebench2024,
  title={LateBench: Benchmark for Late Reasoning Errors in Mathematical Solutions},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/latebench}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Join conversations about mathematical reasoning evaluation
- **Documentation**: Full API docs available in `docs/`

---

**Note**: This tool requires an OpenAI API key and generates content using GPT-4. Be mindful of API usage costs when running large experiments.