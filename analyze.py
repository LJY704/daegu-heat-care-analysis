"""Reproduce exploratory HEAT-Care index and K-means from source_2025_districts.csv.
This is a newly organized reproduction script, not an assertion that it was the
original code used during the competition. Run: pip install -r requirements.txt;
python analyze.py. Requires source CSV in the same folder.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent
raw = pd.read_csv(ROOT / 'source_2025_districts.csv', encoding='utf-8-sig')
assert len(raw) == 9 and raw['district'].nunique() == 9
assert (raw[['population','elderly_population','park_area_m2']] >= 0).all().all()
assert (raw['elderly_population'] <= raw['population']).all()
df = raw.copy()
df['elderly_share_pct'] = 100 * df['elderly_population'] / df['population']
df['park_area_per_person_m2'] = df['park_area_m2'] / df['population']
X = StandardScaler().fit_transform(df[['elderly_share_pct','park_area_per_person_m2']])
df['elderly_z'] = X[:, 0]
df['park_z'] = X[:, 1]
df['exploratory_vulnerability_index'] = (X[:, 0] - X[:, 1]) / 2
for k in (2, 3):
    km = KMeans(n_clusters=k, random_state=42, n_init=50)
    labels = km.fit_predict(X)
    df[f'cluster_k{k}'] = labels + 1  # cluster labels are arbitrary IDs
    print(f'K={k} silhouette: {silhouette_score(X, labels):.3f}')
print(df.sort_values('exploratory_vulnerability_index',ascending=False)[['district','exploratory_vulnerability_index','cluster_k3']].to_string(index=False))
df.sort_values('exploratory_vulnerability_index',ascending=False).to_csv(ROOT/'recomputed_results.csv',index=False,encoding='utf-8-sig')
print('Saved recomputed_results.csv')
