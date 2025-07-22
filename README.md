# LateBench: Educational Mathematical Error Generation

LateBench creates educational examples by injecting subtle logical errors in mathematical solutions. The system generates naturally-written solutions with unnoticeable errors that appear in the final 25% of reasoning steps, designed to test mathematical reasoning capabilities.

## 🎯 Core Purpose

Create mathematical solutions that:
- **Look completely natural** - No hints that errors exist
- **Contain subtle logical flaws** - Not computational mistakes
- **Place errors late** - In the last 25% of solution steps  
- **Test reasoning skills** - Challenge critics and reward models
- **Provide learning value** - Educational error detection practice

## ✨ Key Features

- **Natural Writing**: Solutions read like genuine student work with confident, professional tone
- **5 Error Types**: Logical errors, rule misapplication, invalid generalizations, assumption errors, condition misunderstanding
- **GPT-4 Powered**: Intelligent error injection with context awareness
- **Interactive Dashboard**: Browse, compare, and evaluate examples with GPT-4o-mini critic
- **Educational Focus**: Each error provides valuable learning opportunities

## 🚀 Quick Start

### 1. Setup Environment

```bash
git clone https://github.com/your-username/latebench.git
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

### 3. Test Installation

```bash
python test_system.py
```

### 4. Generate Examples

```bash
python run_experiment.py --experiment small --num_examples 5
```

### 5. Launch Dashboard

```bash
python start_dashboard.py
```

Visit http://localhost:8000 to browse examples interactively.

## 📁 Project Structure

```
latebench/
├── src/                          # Core implementation
│   ├── error_injector.py        # Main error injection system
│   ├── error_types.py           # Error taxonomy definitions
│   ├── critic.py                # GPT-4o-mini evaluation
│   ├── dashboard.py             # Interactive web interface
│   ├── dashboard_utils.py       # Dashboard utilities
│   ├── data_loader.py           # Dataset handling
│   ├── visualization.py         # Analysis tools
│   ├── static/                  # CSS and JavaScript
│   └── templates/               # HTML templates
├── notebooks/                   # Jupyter exploration
├── data/                        # Generated examples and datasets
├── download_data.py             # Dataset download script
├── run_experiment.py            # Batch generation
├── start_dashboard.py           # Dashboard launcher
├── test_system.py               # System validation
└── requirements.txt             # Dependencies
```

## 🎓 Error Types

The system focuses on 5 educationally valuable error categories:

1. **Logical Error**: Incorrect interpretation of conditions, incomplete case analysis
2. **Misunderstanding Conditions**: Using incomplete conditions, misinterpreting requirements  
3. **Incorrect Rules/Properties**: Misapplying theorems outside their valid domain
4. **Invalid Generalizations**: Extending specific cases without justification
5. **Assumption Errors**: Making unjustified assumptions, missing constraints

## 💻 Usage Examples

### Generate Single Example

```python
from src.error_injector import AdversarialErrorInjector
import json

# Load a problem
with open('data/filtered_long_solutions.json', 'r') as f:
    problems = json.load(f)

# Inject error
injector = AdversarialErrorInjector()
result = injector.inject_error(problems[0])

if result.success:
    print(f"Error injected at step {result.error_analysis['selected_error_step']}")
    print(f"Error type: {result.error_analysis['error_type']}")
```

### Batch Generation

```python
# Generate multiple examples
results = injector.batch_inject_errors(problems[:10])
success_rate = sum(1 for r in results if r.success) / len(results)
print(f"Success rate: {success_rate:.1%}")
```

### Critic Evaluation

```python
from src.critic import LLMCritic, evaluate_single_example

# Evaluate an example
critic = LLMCritic(model="gpt-4o-mini")
evaluation = evaluate_single_example(result.modified_solution)
print(f"Errors detected: {evaluation.has_errors}")
```

## 🌐 Interactive Dashboard

The dashboard provides:
- **Four-panel layout**: Problem, original solution, modified solution, critic analysis
- **Example navigation**: Browse through generated examples
- **One-click evaluation**: Run GPT-4o-mini critic on examples
- **Error visualization**: See exactly where errors were placed
- **Export functionality**: Save examples for further analysis

### Dashboard Features

- **Keyboard shortcuts**: Arrow keys for navigation, Ctrl+R for critic evaluation
- **Filtering**: Browse by error type or evaluation status
- **Real-time evaluation**: Test critic performance against known errors
- **Professional UI**: Clean, responsive design for research use

## 📊 Example Output

The system generates examples like:

**Original Problem**: Find the maximum area of triangle ABC given constraints...

**Modified Solution**: Uses confident mathematical language with a subtle logical error in step 6 of 8, leading to an incorrect final answer while appearing completely natural.

**Error Analysis**: Identifies the specific logical flaw and its educational value.

## ⚙️ Configuration

### Environment Variables (.env)

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview  # Optional
REQUESTS_PER_MINUTE=60           # Optional rate limiting
```

### Experiment Configuration

```bash
# Small test run
python run_experiment.py --experiment small --num_examples 5

# Larger dataset
python run_experiment.py --experiment medium --num_examples 50
```

## 🔬 Research Applications

LateBench is designed for:

1. **Educational Assessment**: Train students to detect reasoning errors
2. **Model Evaluation**: Test reasoning critics and reward models
3. **Error Analysis**: Study common logical fallacies in mathematics
4. **Curriculum Development**: Create challenging problem sets

## 📈 Quality Metrics

The system ensures:
- **Natural writing**: No uncertain language or error hints
- **Late placement**: Errors in final 25% of steps
- **Logical consistency**: All steps follow naturally
- **Educational value**: Each error teaches important concepts
- **Answer differentiation**: Modified solutions have different final answers

## 🛠️ Development

### Running Tests

```bash
python test_system.py
```

### Adding Error Types

Extend `src/error_types.py` with new error definitions:

```python
"new_error": ErrorType(
    name="new_error",
    description="Description of the error",
    example="Example case",
    mathematical_context="When this occurs",
    detection_difficulty="High - why it's subtle",
    common_subjects=["algebra", "geometry"]
)
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Submit a pull request

## 📞 Support

- **Issues**: Report bugs on GitHub
- **Documentation**: Full API docs in source code
- **Examples**: Check `data/educational_examples.json` for sample output

---

**Note**: Requires OpenAI API key. Be mindful of API costs when generating large datasets.