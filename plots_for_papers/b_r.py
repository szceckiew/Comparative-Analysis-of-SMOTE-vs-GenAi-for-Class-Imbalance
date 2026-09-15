import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)

X_maj = np.random.normal(loc=[5, 5], scale=1.3, size=(300, 2))

X_min_1 = np.random.normal(loc=[1.5, 5], scale=0.4, size=(8, 2))
X_min_2 = np.random.normal(loc=[8.5, 5], scale=0.4, size=(8, 2))
X_min = np.vstack([X_min_1, X_min_2])

k_bg = 3
nbrs_bg = NearestNeighbors(n_neighbors=k_bg + 1).fit(X_min)
distances_bg, indices_bg = nbrs_bg.kneighbors(X_min)

X_syn_standard = []
for i in range(len(X_min)):
    nn_idx = np.random.choice(indices_bg[i][1:])
    p1_bg = X_min[i]
    p2_bg = X_min[nn_idx]
    
    for _ in range(3):
        lmbda_bg = np.random.uniform(0, 1)
        p_new_bg = p1_bg + lmbda_bg * (p2_bg - p1_bg)
        X_syn_standard.append(p_new_bg)

X_syn_standard = np.array(X_syn_standard)

p1_problem = X_min[0]
p2_problem = X_min[10]

lmbda_problem = 0.5
p_syn_problem = p1_problem + lmbda_problem * (p2_problem - p1_problem)

plt.figure(figsize=(10, 6))

plt.scatter(X_maj[:, 0], X_maj[:, 1], c='#1f77b4', alpha=0.6, label='Majority Class', edgecolors='none', s=60, zorder=1)

plt.scatter(X_syn_standard[:, 0], X_syn_standard[:, 1], c='#ff7f0e', alpha=0.25, s=60, marker='.', zorder=2, label='Synthetic (Intra-cluster)')

plt.scatter(X_min[:, 0], X_min[:, 1], c='#d62728', s=90, label='Minority Class', edgecolors='k', zorder=5)

plt.scatter([p1_problem[0], p2_problem[0]], [p1_problem[1], p2_problem[1]], 
            c='none', edgecolors='black', s=230, linewidths=2.5, zorder=6)

plt.plot([p1_problem[0], p2_problem[0]], [p1_problem[1], p2_problem[1]], 
         'k--', alpha=0.9, linewidth=1.8, label='Erroneous Interpolation Line', zorder=4)

plt.scatter(p_syn_problem[0], p_syn_problem[1], 
            c='#ff7f0e', marker='*', s=500, edgecolors='k', linewidth=1.5, zorder=7, 
            label='Critical Noise Point ($x_{new}$)')

plt.text(p1_problem[0]-0.6, p1_problem[1]+0.5, '$x_i$', fontsize=18, fontweight='bold', zorder=8)
plt.text(p2_problem[0]+0.3, p2_problem[1]+0.5, '$x_{zi}$', fontsize=18, fontweight='bold', zorder=8)

plt.annotate('Overlap / Noise generated\ninside majority region', 
             xy=(p_syn_problem[0], p_syn_problem[1]-0.2), 
             xytext=(p_syn_problem[0], p_syn_problem[1]-2.7),
             arrowprops=dict(facecolor='black', shrink=0.08, width=2.5, headwidth=10, headlength=10),
             fontsize=13, color='darkred', ha='center', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="darkred", lw=2.5))

plt.xlabel('Feature $f_1$', fontsize=15)
plt.ylabel('Feature $f_2$', fontsize=15)

handles, labels = plt.gca().get_legend_handles_labels()
desired_indices = [0, 2, 1, 4, 3]
plt.legend([handles[i] for i in desired_indices], [labels[i] for i in desired_indices], 
           loc='upper right', fontsize=12, framealpha=0.95)

plt.grid(True, linestyle='--', alpha=0.4)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.savefig('smote_problem.png', dpi=300)
plt.savefig('smote_problem.pdf')