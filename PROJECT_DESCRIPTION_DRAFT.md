# LateBench: Mathematical Reasoning Error Detection Framework

## Project Overview

LateBench is a research framework for evaluating mathematical reasoning capabilities by detecting errors in step-by-step problem solutions. The project focuses on creating high-quality datasets with natural error patterns and developing robust evaluation metrics for mathematical reasoning systems. The complexity of data is tested with evaluating LLMs as critics on it.

## Core Objectives

### Primary Goal
Develop a comprehensive benchmark for evaluating AI systems' ability to identify mathematical errors in multi-step reasoning processes, with emphasis on realistic error patterns that occur naturally in mathematical problem-solving.

### Secondary Goals
- Create data processing pipelines for mathematical reasoning datasets
- Establish standardized evaluation metrics compatible with existing benchmarks (ProcessBench, DeltaBench)
- Build tools for automated error injection and quality assessment
- Provide interactive dashboard for dataset exploration and manual validation

## Data Sources & Management

### Source Datasets
- **NumiMath**: Competition-level mathematical problems with step-by-step solutions
- **PRM800K/MATH**: Process reward model training data with step-level annotations on MATH dataset problems
- **ProcessBench/OlympiadBench/OmniMATH**: Mathematical reasoning benchmarks with annotated errors
- **DeltaBench**: Mathematical reasoning benchmark with human annotated errors
- **Custom**: Privately sourced data of mathematical problems with correct solutions.

### Data Processing Pipeline
1. **Ingestion**: Standardize diverse dataset formats into unified LateBench schema
2. **Quality Filtering**: Filter problems for complexity, long complete solutions or late errors (depending on data), remove any formatting issues.
3. **Error Injection**: Generate realistic mathematical errors using LLM-based adversarial methods. The general error injection is automatic for large scale injection. Manual process is based on the same approach, but with custom added human comments on the error. Organization of comments and manual filtering is done via a dashboard.
4. **Validation**: Manual quality check with the dashboard. Critic performance checks on the data as an automated quality check.
5. **Storage**: Organized file structure with metadata tracking and version control

### Data Quality Standards
- Single error per problem to enable clear evaluation, as the task is to track the first problem
- Errors must be mathematically meaningful (not just formatting issues), preferrably logical errors.
- Error placement in latter third of solution steps for realistic difficulty, our paradigm of late errors.
- Maintain original problem difficulty and subject classification, when available

## Error Injection System

### Adversarial Error Injection
- **Method**: LLM-based error generation using LLMs with specialized prompts
- **Error Types**: Available complicated reasoning erros: Incorrect rules/properties, logical flaws, incorrect assumptions, etc. Not just calculation mistakes!
- **Targeting**: Focus on steps requiring deeper mathematical understanding
- **Quality Control** Injected errors should be hard to catch, sounde plausible for the reviewer

### Parallel Processing Architecture
- **Implementation**: ThreadPoolExecutor with 10 parallel workers
- **Rate Limiting**: Respect OpenAI API limits across threads
- **Progress Tracking**: Real-time progress monitoring with tqdm
- **Performance**: Speedup over sequential processing

## Critic System

### LLM-Based Mathematical Error Detection
- **Model**: GPT-4o-mini for cost-effective evaluation at scale
- **Input**: Problem statement + solution steps with an error (injected or naturally sourced) (no data leakage indicating the ground truth)
- **Output**: Depending on the metric, we either ask for a sequence of errored sections, i.e. binary error classification per step + explanations, or a one number prediciton for the FIRST error. The difference is the effective tradeoff between precision/recall in the two approaches. 
- **Prompt Design**: Specialized for mathematical coherence evaluation

### Key Features
- **Data Leakage Prevention**: No access to original solutions or error markers
- **Step-Level Analysis**: Detailed error identification with explanations
- **Rate Limiting**: Sustainable API usage for large-scale evaluation

## Evaluation Metrics

### DeltaBench-Compatible Framework
For this task the critic prediction is a sequence of errored sections, for example [3, 6, 11, 16]
- **Step-Level Metrics**: Precision, recall, F1-score at individual step level
- **First-Error Cutoff**: Only evaluate steps up to first actual error, i.e. if the ground truth error is at step [11], then the [3, 6, 11, 16] prediction is cut to be [3, 6, 11].
- **Micro-Averaging**: Aggregate true positives, false positives across all steps
- **Macro-Averaging**: Per-problem metrics averaged across dataset

### ProcessBench Metrics
For this task the critic prediciton is just one number (first error position)
- **First Error Accuracy**: Exact identification of first error step location
- **Error Detection Accuracy**: Problem-level binary classification performance -- if we caught the first error or not, with corresponding Precision/Recall. 
- **False Positive Rate**: Incorrect error flagging in error-free problems

### Evaluation Example
For DeltaBench metrics
```
Error at step 20, predictions [5, 7, 10, 13, 20, 24]
→ Valid range: [5, 7, 10, 13, 20] (cutoff at step 20)
→ Recall: 1.0 (found step 20)
→ Precision: 0.2 (1 correct out of 5 predictions)
```
For ProcessBench metrics
```
Error at step 20, prediction is [15] -- we get 1 false error, one uncaught error.

Error at step 20, prediction is "no error" -- we get one uncaught error
```
ProcessBench metrics are interesting on the data that has completely correct solutions, which we have in our dataset. For meaningful result, the problems evaluated with this method should be balanced 50/50 as correct/incorrect

## Technical Infrastructure

### Processing Pipeline
- **Batch Injection**: Parallel error injection across 100+ problems
- **Critic Evaluation**: Multi-threaded evaluation with progress tracking
- **Results Storage**: Structured JSON output with timestamp organization
- **Dashboard Interface**: Interactive exploration of datasets and results

### Performance Optimization
- **Parallel Processing**: Thread-safe operations with shared progress tracking
- **Memory Management**: Efficient handling of large datasets
- **API Rate Limiting**: Sustainable usage of external LLM services
- **Caching**: Avoid redundant computations during development

## Current Status & Results

### Latest Evaluation Results
Example of seen results:
- **Dataset**: 100 NumiMath problems with single error injection
- **Critic Performance**: 47% recall, 10.3% precision, 16.9% F1-score DeltaBench style
- **Key Finding**: Critic has high error detection capability but suffers from over-sensitivity
- **Processing Time**: 3 minutes evaluation + 20.5 minutes injection

### Infrastructure Status
- ✅ Parallel batch injection system
- ✅ Step-level DeltaBench evaluation framework
- ✅ Interactive dashboard for dataset exploration
- ✅ Comprehensive error injection quality analysis
- ⏳ Critic sensitivity tuning for production readiness

## Repository Structure

```
latebench/
├── src/
│   ├── data_processing/     # Dataset ingestion and standardization
│   ├── error_injection/     # Adversarial error generation
│   ├── critic/             # LLM-based error detection
│   ├── metrics/            # DeltaBench evaluation framework
│   └── storage/            # Data management utilities
├── dashboard/              # Interactive web interface
├── data/                   # Datasets and results
├── run_batch_injection.py  # Main batch processing script
├── run_critic_on_batch_data.py  # Evaluation pipeline
└── CRITIC_EVALUATION_ANALYSIS.md  # Latest results analysis
```

## Future Directions

### Immediate Priorities
1. **Full organization**: Organize complete pipeline for adding problems into the final dataset. It should be straightforward to work with. Test each part, quality of created samples, and overall critic performance on with respect to different types of erros (natural, injected with general prompt, injected with manual comments).
3. **Critic Baseline**: Establish the best critic performance on the data.
4. **Human Validation**: Expert annotation of critic predictions for calibration

### Medium-Term Goals
1. **Error Type Classification**: Categorize errors by mathematical domain

### Long-Term Vision
1. **Community Benchmark**: Establish as standard evaluation framework
2. **Scale**: Work on scaling, to use as training data for critics
3. **Critic training**: Fine-tune the critic models on the dataset and see if it improves performance.
4. **Educational Applications**: Tools for mathematical education and assessment