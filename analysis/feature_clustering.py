"""
Analyze what clusters together in the feature embeddings.
Examines whether frames from the same game, same teams, same actions cluster together.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
import json
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import torch
from utils import get_dataloaders

def load_metadata_from_labels(labels_dir='data/TACDEC/labels'):
    """
    Load metadata (game_id, teams) from label files.
    Returns list of (clip_id, game_id, team_home_id, team_away_id) tuples in order.
    """
    clips = []
    labels_path = Path(labels_dir)
    
    for label_file in sorted(labels_path.glob("*.json")):
        try:
            with open(label_file, 'r') as f:
                data = json.load(f)
                clip_id = data['id']
                game_id = data['metadata']['game_id']
                team_home_id = data['metadata']['team_home']['id']
                team_away_id = data['metadata']['team_away']['id']
                clips.append((clip_id, game_id, team_home_id, team_away_id))
        except Exception as e:
            print(f"Warning: Could not load {label_file}: {e}")
    
    return clips

def extract_features_with_sample_tracking(loader):
    """
    Extract features while tracking which sample (clip) each belongs to.
    This is tricky since the dataloader gives us batches without explicit sample IDs.
    As an alternative, we return global indices that could be mapped back.
    """
    all_features = []
    all_labels = []
    all_sample_indices = []
    all_frame_indices = []
    
    sample_idx = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            features = batch['features']  # [batch_size, seq_len, 1024]
            labels = batch['labels']      # [batch_size, seq_len]
            mask = batch['mask']          # [batch_size, seq_len]
            
            batch_size = features.shape[0]
            
            for b in range(batch_size):
                feat_b = features[b]  # [seq_len, 1024]
                label_b = labels[b]   # [seq_len]
                mask_b = mask[b]      # [seq_len]
                
                valid_idx = mask_b.bool()
                valid_features = feat_b[valid_idx].numpy()
                valid_labels = label_b[valid_idx].numpy()
                
                all_features.append(valid_features)
                all_labels.append(valid_labels)
                all_sample_indices.extend([sample_idx] * len(valid_features))
                all_frame_indices.extend(list(range(len(valid_features))))
                
                sample_idx += 1
    
    return (np.concatenate(all_features, axis=0), 
            np.concatenate(all_labels, axis=0),
            np.array(all_sample_indices),
            np.array(all_frame_indices))

def analyze_clustering_by_game(features, labels, sample_indices, clips_metadata, model_name="Model"):
    """
    Analyze if frames from the same game cluster together in embedding space.
    """
    print(f"\n{'='*70}")
    print(f"[{model_name}] Clustering Analysis by Game")
    print('='*70)
    
    # Group frames by game
    game_clusters = defaultdict(list)
    for frame_idx, sample_idx in enumerate(sample_indices):
        if sample_idx < len(clips_metadata):
            game_id = clips_metadata[sample_idx][1]
            game_clusters[game_id].append(frame_idx)
    
    print(f"\nTotal frames: {len(features)}")
    print(f"Total games represented: {len(game_clusters)}")
    print(f"Frames per game (avg): {np.mean([len(v) for v in game_clusters.values()]):.1f}")
    
    # For each game, compute intra-game similarity
    intra_game_sims = []
    for game_id, frame_indices in game_clusters.items():
        if len(frame_indices) > 1:
            game_features = features[frame_indices]
            sim_matrix = cosine_similarity(game_features)
            # Exclude diagonal
            np.fill_diagonal(sim_matrix, np.nan)
            intra_sim = np.nanmean(sim_matrix)
            intra_game_sims.append(intra_sim)
    
    if intra_game_sims:
        print(f"\nIntra-game frame similarity (cosine):")
        print(f"  Mean: {np.mean(intra_game_sims):.4f}")
        print(f"  Std:  {np.std(intra_game_sims):.4f}")
        print(f"  Min:  {np.min(intra_game_sims):.4f}")
        print(f"  Max:  {np.max(intra_game_sims):.4f}")
    
    # Sample some cross-game similarities
    game_ids = list(game_clusters.keys())
    cross_game_sims = []
    for i in range(min(10, len(game_ids))):
        game1_frames = game_clusters[game_ids[i]]
        for j in range(i+1, min(10, len(game_ids))):
            game2_frames = game_clusters[game_ids[j]]
            g1_features = features[game1_frames]
            g2_features = features[game2_frames]
            sim = np.mean(cosine_similarity(g1_features, g2_features))
            cross_game_sims.append(sim)
    
    if cross_game_sims:
        print(f"\nInter-game frame similarity (sample):")
        print(f"  Mean: {np.mean(cross_game_sims):.4f}")
        print(f"  Std:  {np.std(cross_game_sims):.4f}")
    
    print(f"\n→ Conclusion: Frames {'DO' if np.mean(intra_game_sims) > np.mean(cross_game_sims) else 'DO NOT'} "
          f"cluster by game (intra-game sim: {np.mean(intra_game_sims):.4f} vs inter-game: {np.mean(cross_game_sims):.4f})")

def analyze_clustering_by_team(features, labels, sample_indices, clips_metadata, model_name="Model"):
    """
    Analyze if frames from the same team (either home or away) cluster together.
    """
    print(f"\n{'='*70}")
    print(f"[{model_name}] Clustering Analysis by Team")
    print('='*70)
    
    # Group frames by team pair (home_id, away_id) or individual team
    team_pair_clusters = defaultdict(list)
    team_individual_clusters = defaultdict(list)
    
    for frame_idx, sample_idx in enumerate(sample_indices):
        if sample_idx < len(clips_metadata):
            clip_id, game_id, team_home, team_away = clips_metadata[sample_idx]
            team_pair = (team_home, team_away)
            team_pair_clusters[team_pair].append(frame_idx)
            team_individual_clusters[team_home].append(frame_idx)
            team_individual_clusters[team_away].append(frame_idx)
    
    print(f"\nTeam pairs represented: {len(team_pair_clusters)}")
    print(f"Individual teams represented: {len(team_individual_clusters)}")
    
    # Analyze by team pair
    intra_pair_sims = []
    for team_pair, frame_indices in team_pair_clusters.items():
        if len(frame_indices) > 1:
            pair_features = features[frame_indices]
            sim_matrix = cosine_similarity(pair_features)
            np.fill_diagonal(sim_matrix, np.nan)
            intra_sim = np.nanmean(sim_matrix)
            intra_pair_sims.append(intra_sim)
    
    if intra_pair_sims:
        print(f"\nIntra-team-pair (same matchup) frame similarity:")
        print(f"  Mean: {np.mean(intra_pair_sims):.4f}")
        print(f"  Std:  {np.std(intra_pair_sims):.4f}")
    
    # Analyze by individual team
    intra_team_sims = []
    for team_id, frame_indices in team_individual_clusters.items():
        if len(frame_indices) > 1:
            team_features = features[frame_indices]
            sim_matrix = cosine_similarity(team_features)
            np.fill_diagonal(sim_matrix, np.nan)
            intra_sim = np.nanmean(sim_matrix)
            intra_team_sims.append(intra_sim)
    
    if intra_team_sims:
        print(f"\nIntra-team (same team across games) frame similarity:")
        print(f"  Mean: {np.mean(intra_team_sims):.4f}")
        print(f"  Std:  {np.std(intra_team_sims):.4f}")
    
    print(f"\n→ Conclusion: Frames from same team pair cluster {'MODERATELY' if np.mean(intra_pair_sims) > 0.5 else 'WEAKLY'}")

def analyze_clustering_by_action(features, labels, model_name="Model"):
    """
    Analyze how well different action classes cluster.
    """
    print(f"\n{'='*70}")
    print(f"[{model_name}] Clustering Analysis by Action Type")
    print('='*70)
    
    class_names = {0: 'Background', 1: 'Tackle - Live', 2: 'Tackle - Replay'}
    
    intra_class_sims = {}
    for class_id, class_name in class_names.items():
        class_mask = (labels == class_id)
        class_features = features[class_mask]
        
        if len(class_features) > 1:
            sim_matrix = cosine_similarity(class_features)
            np.fill_diagonal(sim_matrix, np.nan)
            intra_sim = np.nanmean(sim_matrix)
            intra_class_sims[class_name] = intra_sim
    
    print(f"\nIntra-class similarity by action type:")
    for class_name, sim in intra_class_sims.items():
        print(f"  {class_name}: {sim:.4f}")
    
    # Inter-class similarity
    print(f"\nInter-class similarity (sample):")
    for i, (c1, sim1) in enumerate(intra_class_sims.items()):
        for c2, sim2 in list(intra_class_sims.items())[i+1:]:
            c1_features = features[labels == list(intra_class_sims.keys()).index(c1)]
            c2_features = features[labels == list(intra_class_sims.keys()).index(c2)]
            inter_sim = np.mean(cosine_similarity(c1_features[:500], c2_features[:500]))
            print(f"  {c1} vs {c2}: {inter_sim:.4f}")

if __name__ == "__main__":
    print("Loading metadata...")
    clips_metadata = load_metadata_from_labels('data/TACDEC/labels')
    
    print("\nExtracting DINOv3 features with tracking...")
    loaders_dino = get_dataloaders(batch_size=8, backbone_type='dinov3', backbone_size='large', num_classes=3)
    feat_dino, labels_dino, sample_idx_dino, frame_idx_dino = extract_features_with_sample_tracking(loaders_dino[-1])
    
    analyze_clustering_by_action(feat_dino, labels_dino, model_name="DINOv3-Large")
    analyze_clustering_by_game(feat_dino, labels_dino, sample_idx_dino, clips_metadata, model_name="DINOv3-Large")
    analyze_clustering_by_team(feat_dino, labels_dino, sample_idx_dino, clips_metadata, model_name="DINOv3-Large")
    
    print("\n\nExtracting V-JEPA2 features with tracking...")
    loaders_vjepa = get_dataloaders(batch_size=8, backbone_type='vjepa2', backbone_size='large', num_classes=3)
    feat_vjepa, labels_vjepa, sample_idx_vjepa, frame_idx_vjepa = extract_features_with_sample_tracking(loaders_vjepa[-1])
    
    analyze_clustering_by_action(feat_vjepa, labels_vjepa, model_name="V-JEPA2-Large")
    analyze_clustering_by_game(feat_vjepa, labels_vjepa, sample_idx_vjepa, clips_metadata, model_name="V-JEPA2-Large")
    analyze_clustering_by_team(feat_vjepa, labels_vjepa, sample_idx_vjepa, clips_metadata, model_name="V-JEPA2-Large")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
