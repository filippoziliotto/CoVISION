import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

def analyze_baseline_vs_similarity(scene_id, poses, feats, out_dir, sample_limit=5000):
    """
    Analyze how feature similarity correlates with camera baseline.
    Args:
        scene_id: str
        poses: (N, 4, 4) camera extrinsics [R|t]
        feats: (N, D) normalized feature vectors (after pooling)
        out_dir: output path to save the plot
        sample_limit: max number of random pairs to sample for visualization
    """
    N = feats.shape[0]
    centers = np.array([p[:3, 3] for p in poses])  # camera centers (N, 3)
    all_pairs = []
    
    # Compute baseline distances and feature similarities
    for i in range(N):
        for j in range(i + 1, N):
            baseline = np.linalg.norm(centers[i] - centers[j])
            sim = np.dot(feats[i], feats[j])
            all_pairs.append((baseline, sim))
    
    if not all_pairs:
        print(f"[WARN] No pairs for baseline analysis in {scene_id}")
        return None

    arr = np.array(all_pairs)
    baselines = arr[:, 0]
    sims = arr[:, 1]

    # Subsample for plotting
    if len(baselines) > sample_limit:
        idx = np.random.choice(len(baselines), sample_limit, replace=False)
        baselines = baselines[idx]
        sims = sims[idx]

    # Normalize baselines (optional, makes plots clearer)
    baselines_norm = (baselines - baselines.min()) / (baselines.max() - baselines.min() + 1e-8)

    # Fit regression for visualization
    reg = LinearRegression().fit(baselines_norm.reshape(-1, 1), sims)
    trend = reg.predict(np.linspace(0, 1, 100).reshape(-1, 1))
    corr, _ = pearsonr(baselines, sims)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.scatter(baselines_norm, sims, s=10, alpha=0.5, color="steelblue", label="pairs")
    plt.plot(np.linspace(0, 1, 100), trend, color="darkred", lw=2, label=f"trend (r={corr:.3f})")
    plt.xlabel("Normalized baseline distance")
    plt.ylabel("Feature cosine similarity")
    plt.title(f"Baseline vs. Feature Similarity – {scene_id}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "baseline_vs_similarity.png"))
    plt.close()

    print(f"[INFO] {scene_id}: baseline-similarity correlation r = {corr:.3f}")
    return corr