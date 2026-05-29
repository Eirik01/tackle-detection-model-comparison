import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.patches as mpatches
import random
from pathlib import Path
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from utils import get_dataloaders

def extract_features(loader):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            features = batch['features']
            labels = batch['labels']
            mask = batch['mask']
            
            # Flatten to 1D sequence length
            features = features.view(-1, features.shape[-1])
            labels = labels.view(-1)
            mask = mask.view(-1).bool()
            
            valid_features = features[mask].numpy()
            valid_labels = labels[mask].numpy()
            
            all_features.append(valid_features)
            all_labels.append(valid_labels)
        
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_features, all_labels

def compute_clustering_metrics(features, labels, model_name):
    """
    Compute quantitative clustering quality metrics.
    
    Args:
        features: Feature array [N, D]
        labels: Label array [N]
        model_name: String for printing (e.g., "DINOv3" or "V-JEPA2")
    
    Returns:
        dict with metrics
    """
    print(f"\n{'='*60}")
    print(f"Clustering Metrics: {model_name}")
    print(f"{'='*60}")
    
    # 1. Silhouette Score (range: -1 to 1, higher is better)
    silhouette = silhouette_score(features, labels)
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"  (Range: -1 to 1; higher = better; 0 = overlapping)")
    
    # 2. Calinski-Harabasz Index (ratio of between-cluster to within-cluster variance)
    calinski = calinski_harabasz_score(features, labels)
    print(f"Calinski-Harabasz Index: {calinski:.2f}")
    print(f"  (Higher = better separated clusters)")
    
    # 3. Cosine Similarity Analysis
    unique_labels = np.unique(labels)
    class_names = ['Background', 'Tackle-Live', 'Tackle-Replay']
    
    print(f"\nCosine Similarity Analysis:")
    print(f"  Intra-class (within same class):")
    
    intra_class_sims = []
    for label in unique_labels:
        mask = labels == label
        class_features = features[mask]
        if len(class_features) > 1:
            # Compute pairwise cosine similarities for this class
            sim_matrix = cosine_similarity(class_features)
            # Get upper triangle (exclude diagonal)
            sim_values = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            avg_sim = np.mean(sim_values)
            intra_class_sims.append(avg_sim)
            print(f"    {class_names[label]}: {avg_sim:.4f}")
    
    avg_intra = np.mean(intra_class_sims)
    print(f"    Average: {avg_intra:.4f}")
    
    print(f"  Inter-class (between different classes):")
    
    inter_class_sims = []
    for i in range(len(unique_labels)):
        for j in range(i+1, len(unique_labels)):
            label_i, label_j = unique_labels[i], unique_labels[j]
            features_i = features[labels == label_i]
            features_j = features[labels == label_j]
            
            sim_matrix = cosine_similarity(features_i, features_j)
            avg_sim = np.mean(sim_matrix)
            inter_class_sims.append(avg_sim)
            print(f"    {class_names[label_i]} vs {class_names[label_j]}: {avg_sim:.4f}")
    
    avg_inter = np.mean(inter_class_sims)
    print(f"    Average: {avg_inter:.4f}")
    
    # Separation metric: lower inter-class sim is better
    separation_ratio = avg_intra / avg_inter if avg_inter > 0 else float('inf')
    print(f"\n  Cohesion/Separation Ratio: {separation_ratio:.4f}")
    print(f"    (Higher = more cohesive within classes, more separated between)")
    
    print(f"{'='*60}\n")
    
    return {
        'silhouette': silhouette,
        'calinski': calinski,
        'intra_class': avg_intra,
        'inter_class': avg_inter,
        'separation_ratio': separation_ratio
    }

def load_metadata_mapping(labels_dir='data/TACDEC/labels'):
    """
    Load game_id and team_id from all label files.
    Returns a mapping from video_id to (game_id, team_home_id, team_away_id, team_home_name, team_away_name)
    """
    metadata = {}
    labels_path = Path(labels_dir)
    
    for label_file in labels_path.glob("*.json"):
        try:
            with open(label_file, 'r') as f:
                data = json.load(f)
                clip_id = data['id']  # This is {game_id}_{clip_id}
                game_id = data['metadata']['game_id']
                team_home_id = data['metadata']['team_home']['id']
                team_away_id = data['metadata']['team_away']['id']
                team_home_name = data['metadata']['team_home']['name']
                team_away_name = data['metadata']['team_away']['name']
                
                metadata[clip_id] = {
                    'game_id': game_id,
                    'team_home_id': team_home_id,
                    'team_away_id': team_away_id,
                    'team_home_name': team_home_name,
                    'team_away_name': team_away_name
                }
        except Exception as e:
            print(f"Warning: Could not load metadata from {label_file}: {e}")
    
    return metadata

def extract_features_with_metadata(loader, metadata_map, labels_dir='data/TACDEC/labels'):
    """
    Extract features along with metadata (game_id, team_ids).
    Note: This is an approximation - we assign the same metadata to all frames in a clip.
    """
    all_features = []
    all_labels = []
    all_game_ids = []
    all_team_home_ids = []
    all_team_away_ids = []
    
    labels_path = Path(labels_dir)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            features = batch['features']
            labels = batch['labels']
            mask = batch['mask']
            
            # Flatten to 1D sequence length
            features = features.view(-1, features.shape[-1])
            labels = labels.view(-1)
            mask = mask.view(-1).bool()
            
            valid_features = features[mask].numpy()
            valid_labels = labels[mask].numpy()
            valid_mask_count = mask.sum().item()
            
            all_features.append(valid_features)
            all_labels.append(valid_labels)
            
            # For metadata, we need to map back to the original clip
            # This is an approximation: assign the first metadata we find that matches
            # In a real scenario, you'd track the sample indices through the dataloader
            game_ids = np.zeros(valid_mask_count, dtype=int)
            team_home_ids = np.zeros(valid_mask_count, dtype=int)
            team_away_ids = np.zeros(valid_mask_count, dtype=int)
            
            # Try to infer metadata from available labels (this is approximate)
            for clip_id, meta in metadata_map.items():
                game_ids[:] = meta['game_id']
                team_home_ids[:] = meta['team_home_id']
                team_away_ids[:] = meta['team_away_id']
                break  # In a full implementation, we'd track this properly
            
            all_game_ids.append(game_ids)
            all_team_home_ids.append(team_home_ids)
            all_team_away_ids.append(team_away_ids)
        
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_game_ids = np.concatenate(all_game_ids, axis=0)
    all_team_home_ids = np.concatenate(all_team_home_ids, axis=0)
    all_team_away_ids = np.concatenate(all_team_away_ids, axis=0)
    
    return all_features, all_labels, all_game_ids, all_team_home_ids, all_team_away_ids

def compute_clustering_metrics(features, labels, model_name="Model"):
    print(f"\n[{model_name}] Computing quantitative metrics (High-Dimensional)...")
    
    # 1. Silhouette Score: measures cohesion vs separation [-1, 1], higher is better
    sil_score = silhouette_score(features, labels)
    print(f"  Silhouette Score: {sil_score:.4f}")
    
    # 2. Calinski-Harabasz Index: ratio of between-clusters dispersion to within-cluster dispersion, higher is better
    ch_score = calinski_harabasz_score(features, labels)
    print(f"  Calinski-Harabasz Index: {ch_score:.4f}")
    
    # 3. Similarity Analysis (Cosine Similarity)
    print(f"  Cosine Similarity Analysis:")
    unique_labels = np.unique(labels)
    class_names = {0: 'Background', 1: 'Tackle - Live', 2: 'Tackle - Replay'}
    
    for l in unique_labels:
        l_name = class_names.get(l, str(l))
        class_mask = (labels == l)
        
        class_features = features[class_mask]
        other_features = features[~class_mask]
        
        # Intra-class similarity
        intra_sim_matrix = cosine_similarity(class_features)
        # Exclude self-similarity combinations across the diagonal
        np.fill_diagonal(intra_sim_matrix, np.nan)
        mean_intra = np.nanmean(intra_sim_matrix)
        
        # Inter-class similarity
        if len(other_features) > 0:
            inter_sim_matrix = cosine_similarity(class_features, other_features)
            mean_inter = np.mean(inter_sim_matrix)
        else:
            mean_inter = float('nan')
            
        print(f"    - {l_name}: Intra-class = {mean_intra:.4f}, Inter-class (vs others) = {mean_inter:.4f}")
    print()

def plot_clustering(features_dino, labels_dino, features_vjepa, labels_vjepa, subsample=5000):
    # Subsample majority class (label 0 usually)
    def subsample_balanced(features, labels, n_per_class=1000):
        unique_labels = np.unique(labels)
        idx_to_keep = []
        for l in unique_labels:
            idx = np.where(labels == l)[0]
            if len(idx) > n_per_class:
                idx = np.random.choice(idx, n_per_class, replace=False)
            idx_to_keep.append(idx)
        idx_to_keep = np.concatenate(idx_to_keep)
        return features[idx_to_keep], labels[idx_to_keep]

    features_dino_sub, labels_dino_sub = subsample_balanced(features_dino, labels_dino)
    features_vjepa_sub, labels_vjepa_sub = subsample_balanced(features_vjepa, labels_vjepa)

    # Compute numbers before dimensionality reduction but after subsampling (to save time)
    compute_clustering_metrics(features_dino_sub, labels_dino_sub, model_name="DINOv3-Large")
    compute_clustering_metrics(features_vjepa_sub, labels_vjepa_sub, model_name="V-JEPA2-Large")

    # Dimensionality reduction
    # PCA to 50d first for speed, then TSNE
    print("Computing PCA to 50 dims...")
    pca = PCA(n_components=50)
    
    # DINO
    features_dino_50 = pca.fit_transform(features_dino_sub)
    print("Computing TSNE for DINOv3...")
    tsne_dino = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    dino_2d = tsne_dino.fit_transform(features_dino_50)
    
    # VJEPA
    features_vjepa_50 = pca.fit_transform(features_vjepa_sub)
    print("Computing TSNE for VJEPA2...")
    tsne_vjepa = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    vjepa_2d = tsne_vjepa.fit_transform(features_vjepa_50)

    # Plotting - By Action Class
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    class_names = ['Background', 'Tackle - Live', 'Tackle - Replay']
    colors = ['#cccccc', '#d62728', '#1f77b4'] # Light Gray, Red, Blue
    
    for l_idx, (l_name, color) in enumerate(zip(class_names, colors)):
        # Dino
        mask_dino = (labels_dino_sub == l_idx)
        axes[0].scatter(dino_2d[mask_dino, 0], dino_2d[mask_dino, 1], c=color, label=l_name, alpha=0.6, s=15, edgecolors='none')
        
        # VJEPA
        mask_vjepa = (labels_vjepa_sub == l_idx)
        axes[1].scatter(vjepa_2d[mask_vjepa, 0], vjepa_2d[mask_vjepa, 1], c=color, label=l_name, alpha=0.6, s=15, edgecolors='none')

    axes[0].set_title('DINOv3-Large (1024D)', fontsize=14, pad=10)
    axes[1].set_title('V-JEPA2-Large (1024D)\nSpatio-Temporal', fontsize=14, pad=10)
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend()
        
    plt.suptitle("t-SNE Feature Clustering Comparison (By Action Class)", fontsize=16)
    plt.tight_layout()
    
    # Ensure figures directory exists
    from pathlib import Path
    Path('figures').mkdir(parents=True, exist_ok=True)
    
    plt.savefig('figures/feature_clustering_comparison.pdf', dpi=300, bbox_inches='tight')
    print("Saved figure to figures/feature_clustering_comparison.pdf")

if __name__ == "__main__":
    # Set seeds for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create output directory if it doesn't exist
    output_dir = Path('figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CLS-side comparison is DINOv3-only: only the DINOv3 linear probe consumes
    # CLS features. V-JEPA 2 never produced `*_features.npz` (no CLS token in
    # the encoder; the extractor writes dense `*_dense_w*.npz` instead). Dense
    # comparison lives in feature_clustering_dense.py.
    # DINOv3 CLS files are extracted at 25 FPS (see eval_spatial_centred.py).
    print("Extracting testing features for DINOv3 (CLS, 25 FPS — linear probe input)...")
    # get_dataloaders returns raw per-frame CLS features (one row per frame).
    loaders_dino = get_dataloaders(batch_size=8, backbone_type='dinov3',
                                   backbone_size='large', num_classes=3,
                                   extraction_fps=25.0)
    features_dino, labels_dino = extract_features(loaders_dino[-1])  # test loader
    print(f"  DINOv3: {len(features_dino)} total frames")
    print(f"  Class distribution: {np.bincount(labels_dino.astype(int))}")

    # The line-215 redefinition of compute_clustering_metrics returns None, so
    # this is print-only. The per-class / per-pair similarity breakdown we
    # actually want is logged inside the function body.
    print("\nComputing clustering quality metrics...")
    compute_clustering_metrics(features_dino, labels_dino, "DINOv3-Large")

    # Single-backbone t-SNE (no V-JEPA 2 CLS to compare against).
    print("\nComputing PCA(50) + TSNE...")
    pca = PCA(n_components=50)
    feats_50 = pca.fit_transform(features_dino)
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    pts_2d = tsne.fit_transform(feats_50)

    fig, ax = plt.subplots(figsize=(7, 6))
    class_names = ['Background', 'Tackle - Live', 'Tackle - Replay']
    colors = ['#cccccc', '#d62728', '#1f77b4']
    for i, (name, c) in enumerate(zip(class_names, colors)):
        m = labels_dino == i
        ax.scatter(pts_2d[m, 0], pts_2d[m, 1], c=c, label=name,
                   alpha=0.6, s=15, edgecolors='none')
    ax.set_title("DINOv3-Large CLS features (linear-probe input, 25 FPS)",
                 fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend()
    plt.tight_layout()
    plt.savefig('figures/feature_clustering_dinov3_cls.pdf', dpi=300,
                bbox_inches='tight')
    print("Saved figure to figures/feature_clustering_dinov3_cls.pdf")
