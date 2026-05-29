#!/usr/bin/env python3
"""
Analyze where events are positioned in videos.
Check if they're typically in the middle, start, or end to determine if zero-padding is necessary.
"""

import sys
from pathlib import Path
import json
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Use local labels directory
labels_dir = PROJECT_ROOT / 'data' / 'TACDEC' / 'labels'

def analyze_event_positioning():
    """Analyze event positions relative to video length."""
    
    event_positions = {
        'relative_positions': [],  # 0 = start, 0.5 = middle, 1.0 = end
        'events_in_first_10pct': 0,
        'events_in_last_10pct': 0,
        'events_in_middle_80pct': 0,
        'total_events': 0,
    }
    
    print("="*70)
    print("Event Positioning Analysis")
    print("="*70)
    
    for label_file in sorted(labels_dir.glob('*.json')):
        with open(label_file, 'r') as f:
            data = json.load(f)
        
        num_frames = data['media_attributes']['frame_count']
        
        for event in data['events']:
            frame_center = (event['frame_start'] + event['frame_end']) / 2
            relative_pos = frame_center / num_frames
            
            event_positions['relative_positions'].append(relative_pos)
            event_positions['total_events'] += 1
            
            if relative_pos < 0.1:
                event_positions['events_in_first_10pct'] += 1
            elif relative_pos > 0.9:
                event_positions['events_in_last_10pct'] += 1
            else:
                event_positions['events_in_middle_80pct'] += 1
    
    # Compute statistics
    positions = np.array(event_positions['relative_positions'])
    
    print(f"\nTotal events analyzed: {event_positions['total_events']}")
    print(f"\nEvent distribution:")
    print(f"  In first 10%:   {event_positions['events_in_first_10pct']:4d} ({100*event_positions['events_in_first_10pct']/event_positions['total_events']:.1f}%)")
    print(f"  In middle 80%:  {event_positions['events_in_middle_80pct']:4d} ({100*event_positions['events_in_middle_80pct']/event_positions['total_events']:.1f}%)")
    print(f"  In last 10%:    {event_positions['events_in_last_10pct']:4d} ({100*event_positions['events_in_last_10pct']/event_positions['total_events']:.1f}%)")
    
    print(f"\nPosition statistics (0=start, 0.5=middle, 1.0=end):")
    print(f"  Mean:     {positions.mean():.3f}")
    print(f"  Std:      {positions.std():.3f}")
    print(f"  Min:      {positions.min():.3f}")
    print(f"  Max:      {positions.max():.3f}")
    print(f"  Median:   {np.median(positions):.3f}")
    print(f"  Q1:       {np.percentile(positions, 25):.3f}")
    print(f"  Q3:       {np.percentile(positions, 75):.3f}")
    
    # Frame-level analysis
    print(f"\n{'='*70}")
    print(f"Impact Analysis for V-JEPA2 (16-frame window):")
    print(f"{'='*70}")
    
    # For a 16-frame window starting at frame 0, the window would be [0:16]
    # Center frame would be at index 7, so this covers frames [0:16]
    # For frame 0 with lower-middle center: [0-7:0+9] = [-7:9] → padded + frames [0:9]
    
    print(f"\nWindow strategy used: lower-middle center")
    print(f"  For frame i: window = [i-7 : i+9] (16 frames total)")
    print(f"  Center position in window: index 7/16")
    
    # Check edge cases: videos with events in first/last few frames
    print(f"\nEvent accessibility analysis:")
    print(f"  Events needing frames before frame 0: {sum(1 for p in positions if p < 0.05/16)}")  # Less than first few frames
    print(f"  Events needing frames after last: {sum(1 for p in positions if p > (1 - 0.05/16))}")
    
    # More practical: if a video is 130 frames (typical), frames 0-7 would need padding on left
    # and frames 123-129 would need padding on right
    print(f"\nPractical edge case (typical 130-frame video):")
    print(f"  Frames 0-7 would access padded frames")
    print(f"  Frames 122-129 would access padded frames")
    print(f"  Events in these boundary zones: {sum(1 for p in positions if p < 8/130 or p > (130-8)/130)}")
    print(f"  Percentage: {100*sum(1 for p in positions if p < 8/130 or p > (130-8)/130)/event_positions['total_events']:.2f}%")
    
    print(f"\n{'='*70}")
    print(f"Conclusion:")
    print(f"{'='*70}")
    
    boundary_pct = 100*sum(1 for p in positions if p < 8/130 or p > (130-8)/130)/event_positions['total_events']
    if boundary_pct < 2:
        print(f"✓ Only {boundary_pct:.2f}% of events are in edge zones")
        print(f"  Zero-padding is OPTIONAL - most events are far from boundaries")
        print(f"  You could start from frame 8 (center=frame 8, window=[1:17]) safely")
    else:
        print(f"✗ {boundary_pct:.2f}% of events are in edge zones")
        print(f"  Zero-padding is RECOMMENDED to handle these cases")

if __name__ == '__main__':
    analyze_event_positioning()
