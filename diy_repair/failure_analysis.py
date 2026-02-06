"""
Phase 4: Analysis for Home DIY Repair Q&A Failure Modes
Creates heatmaps and analyzes failure patterns from labeled data
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import DEFAULT_OUTPUT_DIR, FILENAMES, TARGET_FAILURE_RATE
from .failure_labeling import FAILURE_MODE_NAMES


class FailureAnalyzer:
    """Analyzer for failure mode patterns and correlations."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.failure_modes = FAILURE_MODE_NAMES

    def generate_failure_summary(self) -> Dict:
        """Generate summary statistics for failure modes."""
        overall_failure = self.df['overall_failure'].mean()
        summary = {
            'total_samples': len(self.df),
            'overall_failure_rate': overall_failure,
            'overall_success_rate': 1 - overall_failure,
            'target_failure_rate': TARGET_FAILURE_RATE,
            'target_met': overall_failure < TARGET_FAILURE_RATE,
            'failure_mode_rates': {},
            'failure_mode_counts': {},
            'most_common_failures': [],
            'least_common_failures': []
        }
        for mode in self.failure_modes:
            rate = self.df[mode].mean()
            count = self.df[mode].sum()
            summary['failure_mode_rates'][mode] = rate
            summary['failure_mode_counts'][mode] = count
        sorted_modes = sorted(self.failure_modes, key=lambda x: summary['failure_mode_rates'][x], reverse=True)
        summary['most_common_failures'] = sorted_modes[:3]
        summary['least_common_failures'] = sorted_modes[-3:]
        return summary

    def create_failure_heatmap(self, save_path: str = None) -> None:
        """Create a heatmap showing failure mode patterns."""
        if save_path is None:
            save_path = str(DEFAULT_OUTPUT_DIR / "failure_heatmap.png")
        failure_matrix = self.df[self.failure_modes].values
        plt.figure(figsize=(12, 8))
        sns.heatmap(failure_matrix.T,
                    xticklabels=[f"Sample {i+1}" for i in range(len(self.df))],
                    yticklabels=[mode.replace('_', ' ').title() for mode in self.failure_modes],
                    cmap='RdYlBu_r', cbar_kws={'label': 'Failure (1) / Success (0)'}, annot=True, fmt='d')
        plt.title('Failure Mode Heatmap Across All Samples', fontsize=16, fontweight='bold')
        plt.xlabel('Sample ID', fontsize=12)
        plt.ylabel('Failure Modes', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Heatmap saved to {save_path}")

    def create_failure_rate_chart(self, save_path: str = None) -> None:
        """Create a bar chart showing failure rates by mode."""
        if save_path is None:
            save_path = str(DEFAULT_OUTPUT_DIR / "failure_rates.png")
        failure_rates = [self.df[mode].mean() for mode in self.failure_modes]
        mode_labels = [mode.replace('_', ' ').title() for mode in self.failure_modes]
        plt.figure(figsize=(12, 6))
        bars = plt.bar(mode_labels, failure_rates, color='lightcoral', alpha=0.7)
        for bar, rate in zip(bars, failure_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        plt.title('Failure Rates by Mode', fontsize=16, fontweight='bold')
        plt.xlabel('Failure Modes', fontsize=12)
        plt.ylabel('Failure Rate', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Failure rates chart saved to {save_path}")

    def analyze_correlations(self, save_path: str = None) -> pd.DataFrame:
        """Analyze correlations between failure modes."""
        if save_path is None:
            save_path = str(DEFAULT_OUTPUT_DIR / "failure_correlations.png")
        correlation_matrix = self.df[self.failure_modes].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True,
                    xticklabels=[m.replace('_', ' ').title() for m in self.failure_modes],
                    yticklabels=[m.replace('_', ' ').title() for m in self.failure_modes])
        plt.title('Failure Mode Correlations', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Correlation heatmap saved to {save_path}")
        return correlation_matrix

    def identify_failure_patterns(self) -> Dict[str, List[int]]:
        """Identify common failure patterns across samples."""
        patterns = {}
        for idx, row in self.df.iterrows():
            pattern = tuple(row[mode] for mode in self.failure_modes)
            pattern_name = self._pattern_to_name(pattern)
            if pattern_name not in patterns:
                patterns[pattern_name] = []
            patterns[pattern_name].append(idx)
        return dict(sorted(patterns.items(), key=lambda x: len(x[1]), reverse=True))

    def _pattern_to_name(self, pattern: Tuple[int, ...]) -> str:
        """Convert failure pattern tuple to readable name."""
        if all(x == 0 for x in pattern):
            return "No Failures"
        failed_modes = [self.failure_modes[i] for i, x in enumerate(pattern) if x == 1]
        return " + ".join([m.replace('_', ' ').title() for m in failed_modes])

    def generate_detailed_report(self, save_path: str = None) -> Dict:
        """Generate comprehensive failure analysis report."""
        if save_path is None:
            save_path = str(DEFAULT_OUTPUT_DIR / FILENAMES["failure_analysis_report"])
        summary = self.generate_failure_summary()
        correlations = self.df[self.failure_modes].corr()
        patterns = self.identify_failure_patterns()
        report = {
            'summary': summary,
            'correlations': correlations.to_dict(),
            'failure_patterns': {k: len(v) for k, v in patterns.items()},
            'detailed_patterns': patterns,
            'recommendations': self._generate_recommendations(summary, patterns)
        }
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"Detailed analysis report saved to {save_path}")
        return report

    def _generate_recommendations(self, summary: Dict, patterns: Dict) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        most_common = summary['most_common_failures'][0]
        recommendations.append(f"Focus on improving '{most_common.replace('_', ' ')}' - it's the most common failure mode ({summary['failure_mode_rates'][most_common]:.1%} failure rate)")
        if "No Failures" in patterns and len(patterns["No Failures"]) > 0:
            success_rate = len(patterns["No Failures"]) / summary['total_samples']
            recommendations.append(f"Good news: {success_rate:.1%} of samples have no failures - analyze these for best practices")
        if summary['overall_failure_rate'] > 0.5:
            recommendations.append("High overall failure rate suggests need for better prompt engineering or model fine-tuning")
        if not summary.get('target_met', True):
            recommendations.append(f"Final failure rate must be < {summary.get('target_failure_rate', TARGET_FAILURE_RATE):.2%} to meet project success criterion. Run correction phase and re-label, or improve prompts.")
        return recommendations

    def print_summary_report(self) -> None:
        """Print a formatted summary report to console."""
        summary = self.generate_failure_summary()
        print("=" * 60)
        print("FAILURE ANALYSIS SUMMARY REPORT")
        print("=" * 60)
        print(f"Total Samples: {summary['total_samples']}")
        print(f"Overall Success Rate: {summary['overall_success_rate']:.1%}")
        print(f"Overall Failure Rate: {summary['overall_failure_rate']:.1%}")
        n = summary['total_samples']
        max_failures = int(n * summary['target_failure_rate'])
        print(f"Target: Failure rate < {summary['target_failure_rate']:.2%}")
        print(f"With {n} samples: ≤ {max_failures} failed example(s) to pass")
        print(f"Target met: {'Yes' if summary['target_met'] else 'No'}")
        print(f"\nFAILURE MODE BREAKDOWN:")
        print("-" * 40)
        for mode in self.failure_modes:
            rate = summary['failure_mode_rates'][mode]
            count = summary['failure_mode_counts'][mode]
            print(f"{mode.replace('_', ' ').title():25}: {rate:6.1%} ({count:2d}/{summary['total_samples']})")
        print(f"\nMOST PROBLEMATIC AREAS:")
        print("-" * 40)
        for i, mode in enumerate(summary['most_common_failures'], 1):
            rate = summary['failure_mode_rates'][mode]
            print(f"{i}. {mode.replace('_', ' ').title()}: {rate:.1%}")


def run_failure_analysis(data_file: str = None) -> None:
    """Main function to run complete failure analysis."""
    if data_file is None:
        data_file = str(DEFAULT_OUTPUT_DIR / FILENAMES["failure_labeled_csv"])
    output_dir = Path(data_file).resolve().parent
    try:
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} samples from {data_file}")
        analyzer = FailureAnalyzer(df)
        print("\n🔍 GENERATING FAILURE ANALYSIS...")
        analyzer.print_summary_report()
        print("\n📊 CREATING VISUALIZATIONS...")
        analyzer.create_failure_heatmap(str(output_dir / "failure_heatmap.png"))
        analyzer.create_failure_rate_chart(str(output_dir / "failure_rates.png"))
        analyzer.analyze_correlations(str(output_dir / "failure_correlations.png"))
        print("\n📋 GENERATING DETAILED REPORT...")
        report = analyzer.generate_detailed_report(str(output_dir / FILENAMES["failure_analysis_report"]))
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
        print("\n✅ Analysis complete!")
    except FileNotFoundError:
        print(f"Error: {data_file} not found. Please run the failure labeling phase first.")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")


if __name__ == "__main__":
    run_failure_analysis()
