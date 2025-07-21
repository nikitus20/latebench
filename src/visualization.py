"""
Visualization utilities for adversarial examples.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Any
import numpy as np

from error_injector import InjectionResult


class AdversarialExampleVisualizer:
    """Create visualizations for adversarial examples and results."""
    
    def __init__(self):
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_example_visualization(self, result: InjectionResult) -> str:
        """Create a text visualization of the error injection."""
        
        if not result.success:
            return f"Error injection failed: {result.error_message}"
        
        original_problem = result.original_problem
        error_analysis = result.error_analysis
        modified_solution = result.modified_solution
        error_explanation = result.error_explanation
        
        viz = f"""
# Problem: {original_problem.get('problem', 'No problem statement')[:200]}...

## Original Correct Solution:
{original_problem.get('solution', 'No solution')[:500]}...

**Correct Answer:** {original_problem.get('answer', 'No answer')}

---

## Modified Solution with Injected Error:

### Error Analysis:
- **Total Steps:** {error_analysis.get('total_steps', 'Unknown')}
- **Target Range:** Steps {error_analysis.get('target_step_range', 'Unknown')}
- **Error Step:** Step {error_analysis.get('selected_error_step', 'Unknown')}
- **Error Type:** {error_analysis.get('error_type', 'Unknown')}
- **Rationale:** {error_analysis.get('error_rationale', 'No rationale')}

### Modified Steps:
"""
        
        # Add step-by-step breakdown
        steps = modified_solution.get('steps', [])
        for step in steps:
            step_num = step.get('step_num', '?')
            content = step.get('content', 'No content')
            modified = step.get('modified', False)
            is_error = step.get('error', False)
            
            if is_error:
                viz += f"\n**→ Step {step_num} [ERROR INJECTED]:**\n{content}\n"
            elif modified:
                viz += f"\n**→ Step {step_num} [MODIFIED]:**\n{content}\n"
            else:
                viz += f"\nStep {step_num}: {content}\n"
        
        viz += f"""
**Incorrect Final Answer:** {modified_solution.get('final_answer', 'No answer')}

### Error Explanation:
- **What Changed:** {error_explanation.get('what_changed', 'No explanation')}
- **Why It's Wrong:** {error_explanation.get('why_incorrect', 'No explanation')}
- **Detection Difficulty:** {error_explanation.get('detection_difficulty', 'No explanation')}

### Metadata:
- **Model Used:** {result.metadata.get('model_used', 'Unknown')}
- **Original Steps:** {result.metadata.get('num_original_steps', 'Unknown')}
- **Target Range:** {result.metadata.get('last_quarter_range', 'Unknown')}
"""
        
        return viz
    
    def create_batch_statistics_plot(self, results: List[InjectionResult], 
                                   save_path: str = None) -> plt.Figure:
        """Create statistical plots for batch results."""
        
        # Extract data
        success_rate = sum(1 for r in results if r.success) / len(results)
        
        # Error types distribution
        error_types = []
        detection_difficulties = []
        step_positions = []
        
        for result in results:
            if result.success:
                error_types.append(result.error_analysis.get('error_type', 'Unknown'))
                detection_difficulties.append(
                    result.error_explanation.get('detection_difficulty', 'Unknown').split(' - ')[0]
                )
                step_positions.append(result.error_analysis.get('selected_error_step', 0))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Adversarial Error Injection Results (n={len(results)})', fontsize=16)
        
        # Success rate pie chart
        axes[0, 0].pie([success_rate, 1-success_rate], 
                      labels=['Success', 'Failed'], 
                      autopct='%1.1f%%',
                      startangle=90)
        axes[0, 0].set_title(f'Success Rate: {success_rate:.1%}')
        
        # Error types distribution
        if error_types:
            error_type_counts = pd.Series(error_types).value_counts()
            error_type_counts.plot(kind='bar', ax=axes[0, 1], rot=45)
            axes[0, 1].set_title('Error Types Distribution')
            axes[0, 1].set_ylabel('Count')
        
        # Detection difficulty distribution
        if detection_difficulties:
            difficulty_counts = pd.Series(detection_difficulties).value_counts()
            difficulty_counts.plot(kind='bar', ax=axes[1, 0], 
                                 color=['green', 'orange', 'red'])
            axes[1, 0].set_title('Detection Difficulty Distribution')
            axes[1, 0].set_ylabel('Count')
        
        # Step position distribution
        if step_positions:
            axes[1, 1].hist(step_positions, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Error Step Position Distribution')
            axes[1, 1].set_xlabel('Step Number')
            axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        return fig
    
    def create_error_type_analysis(self, results: List[InjectionResult], 
                                 save_path: str = None) -> plt.Figure:
        """Create detailed analysis of error types."""
        
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            print("No successful results to analyze")
            return None
        
        # Extract data
        data = []
        for result in successful_results:
            error_type = result.error_analysis.get('error_type', 'Unknown')
            difficulty = result.error_explanation.get('detection_difficulty', 'Unknown').split(' - ')[0]
            step_num = result.error_analysis.get('selected_error_step', 0)
            total_steps = result.error_analysis.get('total_steps', 0)
            relative_position = step_num / total_steps if total_steps > 0 else 0
            
            data.append({
                'error_type': error_type,
                'difficulty': difficulty,
                'step_position': step_num,
                'relative_position': relative_position,
                'total_steps': total_steps
            })
        
        df = pd.DataFrame(data)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Error Type Analysis', fontsize=16)
        
        # Error type vs difficulty heatmap
        pivot_table = df.pivot_table(index='error_type', columns='difficulty', 
                                   aggfunc='size', fill_value=0)
        if not pivot_table.empty:
            sns.heatmap(pivot_table, annot=True, fmt='d', ax=axes[0, 0], cmap='YlOrRd')
            axes[0, 0].set_title('Error Type vs Detection Difficulty')
        
        # Relative position by error type
        if len(df['error_type'].unique()) > 1:
            sns.boxplot(data=df, x='error_type', y='relative_position', ax=axes[0, 1])
            axes[0, 1].set_title('Error Position by Type (Relative to Solution Length)')
            axes[0, 1].set_ylabel('Relative Position (0=start, 1=end)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Solution length distribution by error type
        if len(df['error_type'].unique()) > 1:
            sns.boxplot(data=df, x='error_type', y='total_steps', ax=axes[1, 0])
            axes[1, 0].set_title('Solution Length by Error Type')
            axes[1, 0].set_ylabel('Total Steps')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Difficulty distribution
        difficulty_counts = df['difficulty'].value_counts()
        axes[1, 1].pie(difficulty_counts.values, labels=difficulty_counts.index, 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Overall Detection Difficulty Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Analysis plot saved to {save_path}")
        
        return fig
    
    def create_quality_metrics_report(self, results: List[InjectionResult]) -> Dict[str, Any]:
        """Generate quality metrics for the adversarial examples."""
        
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {"error": "No successful results to analyze"}
        
        # Basic metrics
        total_attempts = len(results)
        successful_attempts = len(successful_results)
        success_rate = successful_attempts / total_attempts
        
        # Error type diversity
        error_types = [r.error_analysis.get('error_type', 'Unknown') for r in successful_results]
        unique_error_types = len(set(error_types))
        error_type_distribution = pd.Series(error_types).value_counts().to_dict()
        
        # Detection difficulty distribution
        difficulties = []
        for result in successful_results:
            diff = result.error_explanation.get('detection_difficulty', 'Unknown')
            if ' - ' in diff:
                diff = diff.split(' - ')[0]
            difficulties.append(diff)
        
        difficulty_distribution = pd.Series(difficulties).value_counts().to_dict()
        
        # Step position analysis
        step_positions = []
        relative_positions = []
        
        for result in successful_results:
            step_num = result.error_analysis.get('selected_error_step', 0)
            total_steps = result.error_analysis.get('total_steps', 0)
            
            step_positions.append(step_num)
            if total_steps > 0:
                relative_positions.append(step_num / total_steps)
        
        # Quality metrics
        metrics = {
            "overall_metrics": {
                "total_attempts": total_attempts,
                "successful_attempts": successful_attempts,
                "success_rate": success_rate,
                "unique_error_types": unique_error_types,
                "total_error_types_available": len(error_types)
            },
            
            "error_type_metrics": {
                "distribution": error_type_distribution,
                "diversity_score": unique_error_types / len(set(error_types)) if error_types else 0,
                "most_common": max(error_type_distribution.items(), key=lambda x: x[1]) if error_type_distribution else None
            },
            
            "difficulty_metrics": {
                "distribution": difficulty_distribution,
                "high_difficulty_rate": difficulty_distribution.get('High', 0) / successful_attempts if successful_attempts > 0 else 0
            },
            
            "position_metrics": {
                "mean_step_position": np.mean(step_positions) if step_positions else 0,
                "mean_relative_position": np.mean(relative_positions) if relative_positions else 0,
                "std_relative_position": np.std(relative_positions) if relative_positions else 0,
                "last_quarter_compliance": sum(1 for pos in relative_positions if pos >= 0.75) / len(relative_positions) if relative_positions else 0
            }
        }
        
        return metrics
    
    def save_html_report(self, results: List[InjectionResult], 
                        output_path: str, max_examples: int = 10):
        """Create an HTML report with examples and statistics."""
        
        # Generate quality metrics
        metrics = self.create_quality_metrics_report(results)
        
        # Select sample examples
        successful_results = [r for r in results if r.success][:max_examples]
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LateBench Adversarial Examples Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .example {{ border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 5px; }}
        .error {{ background: #ffebee; border-left: 4px solid #f44336; }}
        .success {{ background: #e8f5e8; border-left: 4px solid #4caf50; }}
        .step-error {{ background: #ffcdd2; padding: 5px; margin: 5px 0; border-radius: 3px; }}
        .step-modified {{ background: #fff3e0; padding: 5px; margin: 5px 0; border-radius: 3px; }}
        .step-normal {{ background: #f9f9f9; padding: 5px; margin: 5px 0; border-radius: 3px; }}
        pre {{ white-space: pre-wrap; font-size: 14px; }}
    </style>
</head>
<body>
    <h1>LateBench Adversarial Examples Report</h1>
    
    <h2>Quality Metrics</h2>
    <div class="metric">
        <h3>Overall Performance</h3>
        <p><strong>Success Rate:</strong> {metrics['overall_metrics']['success_rate']:.1%} 
           ({metrics['overall_metrics']['successful_attempts']}/{metrics['overall_metrics']['total_attempts']})</p>
        <p><strong>Error Type Diversity:</strong> {metrics['overall_metrics']['unique_error_types']} unique types used</p>
    </div>
    
    <div class="metric">
        <h3>Error Distribution</h3>
        <ul>
"""
        
        # Add error type distribution
        for error_type, count in metrics['error_type_metrics']['distribution'].items():
            html_content += f"            <li><strong>{error_type}:</strong> {count} examples</li>\n"
        
        html_content += f"""
        </ul>
    </div>
    
    <div class="metric">
        <h3>Quality Indicators</h3>
        <p><strong>Late Position Compliance:</strong> {metrics['position_metrics']['last_quarter_compliance']:.1%} 
           of errors in last 25% of solution</p>
        <p><strong>High Difficulty Rate:</strong> {metrics['difficulty_metrics']['high_difficulty_rate']:.1%} 
           of examples marked as high difficulty</p>
        <p><strong>Average Relative Position:</strong> {metrics['position_metrics']['mean_relative_position']:.2f}
           (0=start, 1=end of solution)</p>
    </div>
    
    <h2>Sample Examples</h2>
"""
        
        # Add example cases
        for i, result in enumerate(successful_results):
            problem = result.original_problem.get('problem', 'No problem')[:300] + "..."
            error_type = result.error_analysis.get('error_type', 'Unknown')
            step_num = result.error_analysis.get('selected_error_step', 'Unknown')
            
            html_content += f"""
    <div class="example success">
        <h3>Example {i+1}: {error_type} Error</h3>
        <p><strong>Problem:</strong> {problem}</p>
        <p><strong>Error Injected at Step:</strong> {step_num}</p>
        <p><strong>Original Answer:</strong> {result.original_problem.get('answer', 'N/A')}</p>
        <p><strong>Modified Answer:</strong> {result.modified_solution.get('final_answer', 'N/A')}</p>
        
        <details>
            <summary>View Error Analysis</summary>
            <div class="metric">
                <p><strong>What Changed:</strong> {result.error_explanation.get('what_changed', 'N/A')}</p>
                <p><strong>Why Incorrect:</strong> {result.error_explanation.get('why_incorrect', 'N/A')}</p>
                <p><strong>Detection Difficulty:</strong> {result.error_explanation.get('detection_difficulty', 'N/A')}</p>
            </div>
        </details>
    </div>
"""
        
        html_content += """
</body>
</html>"""
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report saved to {output_path}")

# Create global instance
VISUALIZER = AdversarialExampleVisualizer()