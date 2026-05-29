#!/usr/bin/env python3
"""
Visualization and comparison tool for multi-seed experiments.

Creates publication-quality figures comparing results across multiple 
experiment runs (e.g., different backbones, hyperparameters, etc.).

Usage:
    python analysis/multi_seed_results.py experiments/multi_seed_dinov3_large_20240101_120000
    python compare_experiments.py --exp1 exp1_dir --exp2 exp2_dir --exp3 exp3_dir
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime


class ExperimentAnalyzer:
    """Analyzes and visualizes multi-seed experiment results."""
    
    def __init__(self, experiment_dir):
        self.exp_dir = Path(experiment_dir)
        self.metrics_file = self.exp_dir / "aggregated_metrics.json"
        
        if not self.metrics_file.exists():
            raise FileNotFoundError(f"No aggregated_metrics.json found in {experiment_dir}")
        
        with open(self.metrics_file, 'r') as f:
            self.data = json.load(f)
    
    def get_exp_name(self):
        """Extract experiment name from directory."""
        return self.exp_dir.name
    
    def plot_overall_accuracy(self, ax=None, color='steelblue'):
        """Plot overall accuracy across seeds."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        acc_data = self.data['validation_metrics'].get('overall_accuracy', {})
        if not acc_data:
            return ax
        
        values = acc_data['values']
        mean = acc_data['mean']
        std = acc_data['std']
        
        seeds = list(range(len(values)))
        ax.scatter(seeds, values, s=100, alpha=0.6, color=color, label='Per-seed')
        ax.axhline(mean, color=color, linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}%')
        ax.fill_between(
            [-0.5, len(values)-0.5],
            mean - std, mean + std,
            alpha=0.2, color=color, label=f'±1 Std: {std:.2f}%'
        )
        
        ax.set_xlabel('Seed', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Overall Accuracy Across Seeds', fontsize=13, fontweight='bold')
        ax.set_xticks(seeds)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim([85, 100])
        
        return ax
    
    def plot_per_class_metrics(self, metric='f1', ax=None):
        """Plot per-class metrics (F1, precision, recall, accuracy)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        per_class = self.data['validation_metrics'].get('per_class', {})
        if not per_class:
            return ax
        
        class_names = sorted(per_class.keys())
        means = []
        stds = []
        
        for class_name in class_names:
            class_data = per_class[class_name].get(metric, {})
            if class_data:
                means.append(class_data['mean'])
                stds.append(class_data['std'])
            else:
                means.append(0)
                stds.append(0)
        
        x = np.arange(len(class_names))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        
        ax.set_ylabel(f'{metric.upper()}', fontsize=12)
        ax.set_xlabel('Class', fontsize=12)
        ax.set_title(f'Per-Class {metric.upper()} Scores', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=15, ha='right')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, mean) in enumerate(zip(bars, means)):
            ax.text(bar.get_x() + bar.get_width()/2, mean + stds[i] + 0.03,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        return ax
    
    def plot_training_convergence(self, ax=None):
        """Plot best validation loss across seeds."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        train_data = self.data['train_metrics'].get('best_val_loss', {})
        if not train_data:
            return ax
        
        values = train_data['values']
        mean = train_data['mean']
        
        seeds = list(range(len(values)))
        ax.scatter(seeds, values, s=100, alpha=0.6, color='coral', label='Per-seed')
        ax.axhline(mean, color='coral', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean:.4f}')
        
        ax.set_xlabel('Seed', fontsize=12)
        ax.set_ylabel('Best Validation Loss', fontsize=12)
        ax.set_title('Training Convergence Across Seeds', fontsize=13, fontweight='bold')
        ax.set_xticks(seeds)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        return ax
    
    def plot_confusion_matrix(self, ax=None, cmap='Blues'):
        """Plot averaged confusion matrix."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))
        
        cm = np.array(self.data['validation_metrics'].get('avg_confusion_matrix', []))
        if cm.size == 0:
            return ax
        
        # Normalize for visualization
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=cm.astype('int'), fmt='d', 
                   cmap=cmap, ax=ax, cbar_kws={'label': 'Proportion'})
        
        class_names = ['Tackle-Live', 'Tackle-Replay', 'Background']
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('Averaged Confusion Matrix (5 Seeds)', fontsize=13, fontweight='bold')
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        
        return ax
    
    def create_summary_figure(self, output_path=None):
        """Create a comprehensive 2x2 summary figure."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{self.get_exp_name()} - Multi-Seed Summary', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        self.plot_overall_accuracy(ax=axes[0, 0])
        self.plot_per_class_metrics('f1', ax=axes[0, 1])
        self.plot_training_convergence(ax=axes[1, 0])
        self.plot_confusion_matrix(ax=axes[1, 1])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {output_path}")
        
        return fig
    
    def create_detailed_metrics_report(self, output_path=None):
        """Create a detailed text report of all metrics."""
        report = []
        report.append("\n" + "="*70)
        report.append(f"DETAILED METRICS REPORT: {self.get_exp_name()}")
        report.append("="*70 + "\n")
        
        # Validation metrics summary
        val_metrics = self.data['validation_metrics']
        
        if 'overall_accuracy' in val_metrics:
            oa = val_metrics['overall_accuracy']
            report.append("OVERALL ACCURACY")
            report.append("-" * 70)
            report.append(f"Mean:  {oa['mean']:.2f}%")
            report.append(f"Std:   {oa['std']:.2f}%")
            report.append(f"Range: {oa['min']:.2f}% - {oa['max']:.2f}%")
            report.append("")
        
        if 'per_class' in val_metrics:
            report.append("PER-CLASS DETAILED BREAKDOWN")
            report.append("-" * 70)
            for class_name in sorted(val_metrics['per_class'].keys()):
                class_data = val_metrics['per_class'][class_name]
                report.append(f"\n{class_name}:")
                for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
                    if metric_name in class_data:
                        m = class_data[metric_name]
                        if metric_name == 'accuracy':
                            report.append(f"  {metric_name:10s}: {m['mean']:6.2f}% ± {m['std']:5.2f}%")
                        else:
                            report.append(f"  {metric_name:10s}: {m['mean']:6.4f} ± {m['std']:6.4f}")
            report.append("")
        
        report_text = "\n".join(report) + "\n"
        print(report_text)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Detailed report saved to: {output_path}")
        
        return report_text


def compare_experiments(exp_dirs, output_dir=None):
    """Compare metrics across multiple experiments."""
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all experiments
    experiments = []
    for exp_dir in exp_dirs:
        try:
            analyzer = ExperimentAnalyzer(exp_dir)
            experiments.append(analyzer)
            print(f"✓ Loaded: {analyzer.get_exp_name()}")
        except Exception as e:
            print(f"✗ Failed to load {exp_dir}: {e}")
    
    if not experiments:
        print("No valid experiments loaded!")
        return
    
    # Create comparison figure
    fig, axes = plt.subplots(1, len(experiments), figsize=(5*len(experiments), 5))
    if len(experiments) == 1:
        axes = [axes]
    
    for i, exp in enumerate(experiments):
        exp.plot_overall_accuracy(ax=axes[i])
        axes[i].set_title(exp.get_exp_name(), fontsize=11, fontweight='bold')
    
    plt.suptitle('Overall Accuracy Comparison Across Experiments', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_dir:
        output_path = output_dir / "comparison_accuracy.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison figure saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize multi-seed experiment results"
    )
    
    parser.add_argument('experiment_dir', nargs='?', type=str,
                       help='Path to experiment directory with aggregated_metrics.json')
    parser.add_argument('--compare', nargs='+', type=str,
                       help='Compare multiple experiments')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for figures and reports')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip interactive plotting')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare mode
        compare_experiments(args.compare, args.output_dir)
    elif args.experiment_dir:
        # Single experiment mode
        analyzer = ExperimentAnalyzer(args.experiment_dir)
        
        output_dir = Path(args.output_dir) if args.output_dir else analyzer.exp_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figures
        summary_fig_path = output_dir / "summary_figure.png"
        analyzer.create_summary_figure(output_path=summary_fig_path)
        
        detailed_report_path = output_dir / "detailed_metrics.txt"
        analyzer.create_detailed_metrics_report(output_path=detailed_report_path)
        
        if not args.no_plot:
            plt.show()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
