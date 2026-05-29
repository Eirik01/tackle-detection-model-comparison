"""
Check class distribution in TACDEC dataset splits.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import torch
import numpy as np
from collections import Counter
from utils import get_dataloaders

def check_distribution():
    print("="*60)
    print("TACDEC Dataset Class Distribution Analysis")
    print("="*60)
    
    # Get dataloaders with same split as training
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=1)
    
    splits = {
        'Train': train_loader,
        'Validation': val_loader,
        'Test': test_loader
    }
    
    label_names = {
        0: "Tackle-Live",
        1: "Tackle-Replay", 
        2: "Live-Incomplete",
        3: "Replay-Incomplete",
        4: "Background"
    }
    
    for split_name, loader in splits.items():
        print(f"\n{split_name} Set ({len(loader.dataset)} clips)")
        print("-"*60)
        
        # Collect all labels
        all_labels = []
        for batch in loader:
            labels = batch['labels'].numpy().flatten()
            mask = batch['mask'].numpy().flatten()
            # Only count valid (non-padded) frames
            valid_labels = labels[mask == 1.0]
            all_labels.extend(valid_labels)
        
        # Count occurrences
        label_counts = Counter(all_labels)
        total_frames = len(all_labels)
        
        print(f"Total frames: {total_frames:,}")
        print()
        
        # Print distribution
        for label_id in sorted(label_counts.keys()):
            count = label_counts[label_id]
            percentage = (count / total_frames) * 100
            name = label_names.get(int(label_id), f"Unknown-{label_id}")
            print(f"{name:20s}: {count:6,} frames ({percentage:5.2f}%)")
        
        # Check for missing classes
        missing_classes = set(range(5)) - set(label_counts.keys())
        if missing_classes:
            print()
            print("⚠️  Missing classes:")
            for label_id in sorted(missing_classes):
                name = label_names.get(label_id, f"Unknown-{label_id}")
                print(f"   {name}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    check_distribution()
