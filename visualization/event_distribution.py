#!/usr/bin/env python3
"""
Visualize event positioning distribution within videos for thesis.
Produces publication-quality figures showing temporal event distribution.
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Use local labels directory
labels_dir = PROJECT_ROOT / 'data' / 'TACDEC' / 'labels'

# Set style for thesis
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def analyze_and_visualize():
    """Analyze event positions and create visualizations."""
    
    event_positions = []
    event_times_seconds = []
    event_types = []
    video_lengths = []
    events_by_video = {}
    
    print("Loading TACDEC dataset...")
    
    for label_file in sorted(labels_dir.glob('*.json')):
        with open(label_file, 'r') as f:
            data = json.load(f)
        
        num_frames = data['media_attributes']['frame_count']
        frame_rate = data['media_attributes']['frame_rate']
        video_id = data['id']
        video_lengths.append(num_frames)
        events_by_video[video_id] = []
        
        for event in data['events']:
            frame_center = (event['frame_start'] + event['frame_end']) / 2
            relative_pos = frame_center / num_frames
            time_seconds = frame_center / frame_rate
            event_type = event['type']
            
            event_positions.append(relative_pos)
            event_times_seconds.append(time_seconds)
            event_types.append(event_type)
            events_by_video[video_id].append({
                'relative_pos': relative_pos,
                'time_seconds': time_seconds,
                'type': event_type,
                'frame_center': frame_center,
                'frame_start': event['frame_start'],
                'frame_end': event['frame_end']
            })
    
    event_positions = np.array(event_positions)
    video_lengths = np.array(video_lengths)
    
    print(f"Total videos: {len(events_by_video)}")
    print(f"Total events: {len(event_positions)}")
    print(f"Average video length: {video_lengths.mean():.0f} frames")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Histogram with KDE
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(event_positions, bins=30, alpha=0.7, color='steelblue', edgecolor='black', density=True)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(event_positions)
    x_range = np.linspace(0, 1, 200)
    ax1.plot(x_range, kde(x_range), 'r-', linewidth=2.5, label='KDE')
    ax1.axvline(event_positions.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {event_positions.mean():.3f}')
    ax1.axvline(np.median(event_positions), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(event_positions):.3f}')
    ax1.set_xlabel('Relative Position in Video (0=start, 1=end)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax1.set_title('Event Position Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Boxplot by deciles
    ax2 = plt.subplot(2, 3, 2)
    deciles = []
    decile_labels = []
    for i in range(10):
        mask = (event_positions >= i/10) & (event_positions < (i+1)/10)
        count = mask.sum()
        deciles.append(count)
        decile_labels.append(f'{i*10}-{(i+1)*10}%')
    
    colors = plt.cm.viridis(np.linspace(0, 1, 10))
    bars = ax2.bar(decile_labels, deciles, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Video Position (Deciles)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Events', fontsize=11, fontweight='bold')
    ax2.set_title('Events by Video Decile', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)
    
    # 3. Event type distribution
    ax3 = plt.subplot(2, 3, 3)
    event_type_counts = {}
    event_type_positions = {}
    event_type_times = {}
    for evt_type in set(event_types):
        mask = np.array(event_types) == evt_type
        event_type_counts[evt_type] = mask.sum()
        event_type_positions[evt_type] = event_positions[mask]
        event_type_times[evt_type] = np.array(event_times_seconds)[mask]
    
    colors_types = {'tackle-live': '#1f77b4', 'tackle-replay': '#ff7f0e', 'tackle-live-incomplete': '#2ca02c', 'tackle-replay-incomplete': '#d62728'}
    type_names = sorted(event_type_counts.keys())
    counts = [event_type_counts[t] for t in type_names]
    bar_colors = [colors_types.get(t, '#999999') for t in type_names]
    
    bars = ax3.bar(range(len(type_names)), counts, color=bar_colors, edgecolor='black', linewidth=1.2)
    ax3.set_xticks(range(len(type_names)))
    ax3.set_xticklabels(type_names, rotation=45, ha='right', fontsize=10)
    ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax3.set_title('Event Type Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Violin plot by event type
    ax4 = plt.subplot(2, 3, 4)
    type_names_sorted = sorted(event_type_positions.keys())
    positions_by_type = [event_type_positions[t] for t in type_names_sorted]
    
    parts = ax4.violinplot(positions_by_type, positions=range(len(type_names_sorted)), 
                           showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#8dd3c7')
        pc.set_alpha(0.7)
    
    ax4.set_xticks(range(len(type_names_sorted)))
    ax4.set_xticklabels(type_names_sorted, rotation=45, ha='right', fontsize=10)
    ax4.set_ylabel('Relative Position in Video', fontsize=11, fontweight='bold')
    ax4.set_title('Event Position by Type', fontsize=12, fontweight='bold')
    ax4.set_ylim(-0.05, 1.05)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Cumulative distribution
    ax5 = plt.subplot(2, 3, 5)
    sorted_positions = np.sort(event_positions)
    cumulative = np.arange(1, len(sorted_positions) + 1) / len(sorted_positions)
    ax5.plot(sorted_positions, cumulative, linewidth=2.5, color='darkblue')
    ax5.fill_between(sorted_positions, cumulative, alpha=0.3, color='steelblue')
    ax5.set_xlabel('Relative Position in Video', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax5.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Add reference lines for quartiles
    for q in [0.25, 0.5, 0.75]:
        val = np.quantile(event_positions, q)
        ax5.axvline(val, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
        ax5.text(val, 0.05, f'Q{int(q*4)}\n{val:.2f}', fontsize=9, ha='center')
    
    # 6. Statistics text box
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
EVENT POSITION STATISTICS
{'─' * 40}

Total Events: {len(event_positions)}
Total Videos: {len(events_by_video)}

POSITION STATISTICS:
  Mean:     {event_positions.mean():.4f}
  Median:   {np.median(event_positions):.4f}
  Std Dev:  {event_positions.std():.4f}
  Min:      {event_positions.min():.4f}
  Max:      {event_positions.max():.4f}

QUARTILES:
  Q1 (25%): {np.quantile(event_positions, 0.25):.4f}
  Q2 (50%): {np.quantile(event_positions, 0.50):.4f}
  Q3 (75%): {np.quantile(event_positions, 0.75):.4f}

DISTRIBUTIONS:
  First 10%:    {(event_positions < 0.1).sum():3d} events ({(event_positions < 0.1).sum()/len(event_positions)*100:5.1f}%)
  Middle 80%:   {((event_positions >= 0.1) & (event_positions < 0.9)).sum():3d} events ({((event_positions >= 0.1) & (event_positions < 0.9)).sum()/len(event_positions)*100:5.1f}%)
  Last 10%:     {(event_positions >= 0.9).sum():3d} events ({(event_positions >= 0.9).sum()/len(event_positions)*100:5.1f}%)

VIDEO LENGTH:
  Mean:     {video_lengths.mean():.0f} frames
  Median:   {np.median(video_lengths):.0f} frames
  Min:      {video_lengths.min():.0f} frames
  Max:      {video_lengths.max():.0f} frames
"""
    
    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent / 'results' / 'event_distribution_analysis.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
    print(f"\n✓ Saved: {output_path}")
    
    # Create a second figure: temporal heatmap of event density
    fig2, ax = plt.subplots(figsize=(14, 8))
    
    # Create bins and count events
    num_bins = 100
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    counts, _ = np.histogram(event_positions, bins=bin_edges)
    
    # Create bar chart
    colors_gradient = plt.cm.RdYlGn_r(counts / counts.max())
    bars = ax.bar(bin_centers, counts, width=1/num_bins, color=colors_gradient, edgecolor='none', alpha=0.8)
    
    # Add KDE overlay
    kde = gaussian_kde(event_positions)
    kde_vals = kde(bin_centers) * len(event_positions) * (1/num_bins)
    ax.plot(bin_centers, kde_vals, 'b-', linewidth=2.5, label='KDE (smoothed)', alpha=0.7)
    
    # Styling
    ax.set_xlabel('Position in Video (0=start, 1=end)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Events', fontsize=13, fontweight='bold')
    ax.set_title('Temporal Distribution of Events Within Videos', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=11)
    
    # Add zones
    ax.axvspan(0, 0.1, alpha=0.1, color='red', label='Early zone (0-10%)')
    ax.axvspan(0.9, 1.0, alpha=0.1, color='red', label='Late zone (90-100%)')
    
    plt.tight_layout()
    output_path2 = Path(__file__).parent / 'results' / 'event_temporal_heatmap.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight', format='png')
    print(f"✓ Saved: {output_path2}")
    
    # Create figure showing by event type over time
    fig3, axes = plt.subplots(len(type_names_sorted), 1, figsize=(14, 3*len(type_names_sorted)))
    if len(type_names_sorted) == 1:
        axes = [axes]
    
    max_seconds = 32
    
    for idx, evt_type in enumerate(type_names_sorted):
        ax = axes[idx]
        times = event_type_times[evt_type]
        
        # Filter to only events within max_seconds
        times_filtered = times[times <= max_seconds]
        
        # Histogram with time axis
        counts, bins = np.histogram(times_filtered, bins=50, range=(0, max_seconds))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        colors_grad = plt.cm.viridis(counts / counts.max())
        ax.bar(bin_centers, counts, width=(max_seconds/50), color=colors_grad, edgecolor='black', linewidth=0.5, alpha=0.8)
        
        # KDE
        if len(times_filtered) > 1:
            kde = gaussian_kde(times_filtered)
            kde_x = np.linspace(0, max_seconds, 200)
            kde_vals = kde(kde_x) * len(times_filtered) * (max_seconds/50)
            ax.plot(kde_x, kde_vals, 'r-', linewidth=2.5, label='KDE', alpha=0.8)
        
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title(f'{evt_type} (n={len(times_filtered)})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        if idx == len(type_names_sorted) - 1:
            ax.set_xlabel('Time in Video (seconds)', fontsize=11, fontweight='bold')
        ax.set_xlim(0, max_seconds)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    output_path3 = Path(__file__).parent / 'results' / 'event_by_type_temporal.png'
    plt.savefig(output_path3, dpi=300, bbox_inches='tight', format='png')
    print(f"✓ Saved: {output_path3}")
    
    print("\n" + "="*70)
    print("All visualizations created successfully!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  1. {output_path.name} - Comprehensive 6-panel analysis")
    print(f"  2. {output_path2.name} - Detailed temporal heatmap")
    print(f"  3. {output_path3.name} - Distribution by event type")
    print(f"\nAll files saved to: {output_path.parent}")

if __name__ == '__main__':
    analyze_and_visualize()
