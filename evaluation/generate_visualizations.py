"""
evaluation/generate_visualizations.py
Generate publication-quality graphs for the paper.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

# Set publication-quality style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['grid.color'] = '#e0e0e0'

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. ROUGE METRICS COMPARISON (Bar Chart)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rouge_comparison():
    """Generate ROUGE-1/2/L comparison across models."""
    models = ['TextRank', 'BART', 'HLA-MMR']
    rouge1 = [0.2697, 0.4578, 0.1724]
    rouge2 = [0.1462, 0.2399, 0.0544]
    rougel = [0.2471, 0.3220, 0.1222]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, rouge1, width, label='ROUGE-1', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x, rouge2, width, label='ROUGE-2', color='#A23B72', alpha=0.8)
    bars3 = ax.bar(x + width, rougel, width, label='ROUGE-L', color='#F18F01', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROUGE Score', fontsize=12, fontweight='bold')
    ax.set_title('ROUGE Metrics Comparison Across Summarization Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(0, 0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_rouge_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_rouge_comparison.pdf', bbox_inches='tight')
    print("✓ Generated: fig_rouge_comparison.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 2. FACTUAL CONSISTENCY (SummaC) COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def plot_summac_comparison():
    """Generate SummaC factual consistency comparison."""
    models = ['TextRank', 'BART', 'HLA-MMR']
    summac = [0.5072, 0.4797, -0.0192]
    colors = ['#06A77D' if s > 0.4 else '#D62828' if s < 0 else '#F77F00' for s in summac]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(models, summac, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('SummaC Score', fontsize=12, fontweight='bold')
    ax.set_title('Factual Consistency Comparison (SummaC Metric)', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.2, 0.6)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, summac):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}',
               ha='center', va='bottom' if val > 0 else 'top', fontsize=11, fontweight='bold')
    
    # Add legend
    high_patch = mpatches.Patch(color='#06A77D', label='High Consistency (>0.4)', alpha=0.8)
    med_patch = mpatches.Patch(color='#F77F00', label='Medium Consistency', alpha=0.8)
    low_patch = mpatches.Patch(color='#D62828', label='Low/Negative Consistency', alpha=0.8)
    ax.legend(handles=[high_patch, med_patch, low_patch], fontsize=10, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_summac_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_summac_comparison.pdf', bbox_inches='tight')
    print("✓ Generated: fig_summac_comparison.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3. MULTI-METRIC HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_heatmap():
    """Generate heatmap of all metrics across models."""
    data = {
        'TextRank': [0.2697, 0.1462, 0.2471, 0.5072],
        'BART': [0.4578, 0.2399, 0.3220, 0.4797],
        'HLA-MMR': [0.1724, 0.0544, 0.1222, -0.0192],
    }
    
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'SummaC']
    df = pd.DataFrame(data, index=metrics).T
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Manual heatmap using imshow
    df_norm = df.copy()
    df_norm['SummaC'] = (df_norm['SummaC'] + 0.1) / 0.6
    
    im = ax.imshow(df_norm.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(df)))
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_yticklabels(df.index, fontsize=11)
    
    # Add text annotations
    for i in range(len(df)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{df.iloc[i, j]:.4f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    ax.set_title('Performance Metrics Heatmap Across Models', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Score', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_metrics_heatmap.pdf', bbox_inches='tight')
    print("✓ Generated: fig_metrics_heatmap.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. MULTI-FORMAT PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_multiformat_performance():
    """Generate multi-format input performance comparison."""
    formats = ['PDF', 'DOCX', 'Image']
    summary_lengths = [71, 85, 81]
    colors_fmt = ['#264653', '#2A9D8F', '#E9C46A']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(formats, summary_lengths, color=colors_fmt, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Summary Length (words)', fontsize=12, fontweight='bold')
    ax.set_title('Multi-Format Input Processing Performance', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, summary_lengths):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val} words',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add observations as text
    observations = [
        'Accurate extraction\nand summarization',
        'Well-structured,\npreserves formatting',
        'OCR-based,\nquality dependent'
    ]
    
    for i, (bar, obs) in enumerate(zip(bars, observations)):
        ax.text(bar.get_x() + bar.get_width()/2., 5,
               obs,
               ha='center', va='bottom', fontsize=9, style='italic', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_multiformat_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_multiformat_performance.pdf', bbox_inches='tight')
    print("✓ Generated: fig_multiformat_performance.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL COMPARISON RADAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_radar_comparison():
    """Generate radar chart comparing all models across metrics."""
    categories = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'SummaC']
    
    textrank = [0.2697, 0.1462, 0.2471, 0.5072]
    bart = [0.4578, 0.2399, 0.3220, 0.4797]
    hla_mmr = [0.1724, 0.0544, 0.1222, 0.0]  # Clamp negative to 0 for radar
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    textrank += textrank[:1]
    bart += bart[:1]
    hla_mmr += hla_mmr[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, textrank, 'o-', linewidth=2, label='TextRank', color='#2E86AB')
    ax.fill(angles, textrank, alpha=0.25, color='#2E86AB')
    
    ax.plot(angles, bart, 'o-', linewidth=2, label='BART', color='#A23B72')
    ax.fill(angles, bart, alpha=0.25, color='#A23B72')
    
    ax.plot(angles, hla_mmr, 'o-', linewidth=2, label='HLA-MMR', color='#F18F01')
    ax.fill(angles, hla_mmr, alpha=0.25, color='#F18F01')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 0.5)
    ax.set_title('Multi-Metric Model Comparison (Radar Chart)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_radar_comparison.pdf', bbox_inches='tight')
    print("✓ Generated: fig_radar_comparison.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 6. PERFORMANCE TRADE-OFF ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def plot_tradeoff_analysis():
    """Generate scatter plot showing quality vs efficiency trade-off."""
    models = ['TextRank', 'BART', 'HLA-MMR']
    quality = [0.2697, 0.4578, 0.1724]  # ROUGE-1 as quality metric
    efficiency = [100, 10, 30]  # Relative speed (TextRank=100, BART=10, HLA-MMR=30)
    colors_trade = ['#06A77D', '#D62828', '#F77F00']
    sizes = [300, 400, 350]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (model, q, e, color, size) in enumerate(zip(models, quality, efficiency, colors_trade, sizes)):
        ax.scatter(e, q, s=size, alpha=0.7, color=color, edgecolors='black', linewidth=2, label=model)
        ax.annotate(model, (e, q), fontsize=11, fontweight='bold', 
                   ha='center', va='center', color='white')
    
    ax.set_xlabel('Relative Speed (higher = faster)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Quality (ROUGE-1)', fontsize=12, fontweight='bold')
    ax.set_title('Quality vs Efficiency Trade-off Analysis', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 0.5)
    ax.grid(True, alpha=0.3)
    
    # Add quadrant labels
    ax.text(90, 0.45, 'Fast & High Quality\n(Ideal)', fontsize=10, style='italic', 
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.text(20, 0.45, 'Slow & High Quality\n(Accurate)', fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_tradeoff_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_tradeoff_analysis.pdf', bbox_inches='tight')
    print("✓ Generated: fig_tradeoff_analysis.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 7. COMBINED METRICS COMPARISON (Grouped Bar Chart)
# ─────────────────────────────────────────────────────────────────────────────

def plot_combined_metrics():
    """Generate comprehensive metrics comparison."""
    models = ['TextRank', 'BART', 'HLA-MMR']
    rouge1 = [0.2697, 0.4578, 0.1724]
    summac = [0.5072, 0.4797, -0.0192]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, rouge1, width, label='ROUGE-1', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, summac, width, label='SummaC', color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('ROUGE-1 vs SummaC: Quality vs Consistency Trade-off', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylim(-0.1, 0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_combined_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_combined_metrics.pdf', bbox_inches='tight')
    print("✓ Generated: fig_combined_metrics.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
    print("=" * 70)
    
    plot_rouge_comparison()
    plot_summac_comparison()
    plot_metrics_heatmap()
    plot_multiformat_performance()
    plot_radar_comparison()
    plot_tradeoff_analysis()
    plot_combined_metrics()
    
    print("\n" + "=" * 70)
    print(f"✓ All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. fig_rouge_comparison.png/pdf - ROUGE metrics comparison")
    print("  2. fig_summac_comparison.png/pdf - Factual consistency comparison")
    print("  3. fig_metrics_heatmap.png/pdf - All metrics heatmap")
    print("  4. fig_multiformat_performance.png/pdf - Multi-format performance")
    print("  5. fig_radar_comparison.png/pdf - Radar chart comparison")
    print("  6. fig_tradeoff_analysis.png/pdf - Quality vs efficiency trade-off")
    print("  7. fig_combined_metrics.png/pdf - ROUGE-1 vs SummaC comparison")
