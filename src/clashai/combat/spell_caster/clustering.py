# clashai/combat/spell_caster/clustering.py
# Distance-based (BFS) clustering — keeps the member points per cluster.
#
# NB: distinct from combat_observer._cluster_positions — this variant also
# returns the 'points' list (used by spell targeting), so it is kept separate.

import numpy as np


def cluster_positions(positions, min_cluster_size=2, cluster_radius=150):
    """
    Groups nearby positions into clusters.

    Returns:
        clusters: list of {'center': (x,y), 'size': n, 'points': [...]}
                  sorted by descending size
    """
    if not positions:
        return []

    points = np.array(positions, dtype=float)
    visited = [False] * len(points)
    clusters = []

    for i in range(len(points)):
        if visited[i]:
            continue

        cluster_pts = [i]
        visited[i] = True
        queue = [i]

        while queue:
            current = queue.pop(0)
            for j in range(len(points)):
                if visited[j]:
                    continue
                dist = np.linalg.norm(points[current] - points[j])
                if dist < cluster_radius:
                    visited[j] = True
                    cluster_pts.append(j)
                    queue.append(j)

        if len(cluster_pts) >= min_cluster_size:
            cluster_points = points[cluster_pts]
            center = cluster_points.mean(axis=0)
            clusters.append({
                'center': (int(center[0]), int(center[1])),
                'size': len(cluster_pts),
                'points': cluster_points.tolist(),
            })

    clusters.sort(key=lambda c: c['size'], reverse=True)
    return clusters
