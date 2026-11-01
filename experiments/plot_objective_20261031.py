"""Render only aggregate, source-locked objective results; no model/data access."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

path = Path(__file__).with_name('context-variance-results-20261031.json')
d = json.loads(path.read_text())
fig, ax = plt.subplots(figsize=(9, 5.3))
groups = [('prediction', 'Prediction only', '#2166ac'),
          ('context_variance', 'With variance penalty', '#b2182b'),
          ('random', 'Shared random encoder', '#737373'), ('pixels', 'Shared pixels', '#1b7837')]
x = np.arange(3)
for i, (key, name, color) in enumerate(groups):
    values = [100*d['summary'][key][head]['confirmation']['mean_accuracy'] for head in ('adamw', 'ridge', 'knn')]
    bars = ax.bar(x + (i-1.5)*.2, values, .18, label=name, color=color)
    ax.bar_label(bars, fmt='%.1f', fontsize=9, padding=3)
ax.set_xticks(x, ['AdamW linear probe', 'Ridge linear probe', 'Cosine kNN'])
ax.set_ylabel('Accuracy on 500 reserved validation examples (%)')
ax.set_ylim(0, 40)
ax.set_axisbelow(True)
ax.grid(axis='y', alpha=.18)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('Context-variance objective: frozen CIFAR-10 evaluation', loc='left', pad=42, fontsize=14)
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=9)
fig.text(.1, .025, 'Matched encoder seeds and shared examples; descriptive means, not independent confidence intervals.\nOfficial test unused. The variance penalty lowered the primary result by 7.07 percentage points.', fontsize=9)
fig.tight_layout(rect=(0, .09, 1, 1))
folder = path.parent/'figures'
folder.mkdir(exist_ok=True)
fig.savefig(folder/'context-variance-20261031.svg')
fig.savefig(folder/'context-variance-20261031.png', dpi=150)
