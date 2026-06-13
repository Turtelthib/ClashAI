# clashai/combat/combat_observer/clustering.py
# Distance-based (BFS) clustering of battlefield positions.

import numpy as np

from clashai.combat.combat_observer.constants import CLUSTER_RADIUS, MIN_CLUSTER_SIZE


def _cluster_positions(positions, radius=CLUSTER_RADIUS, min_size=MIN_CLUSTER_SIZE):
    """
    Simple distance-based clustering (BFS).

    Returns:
        clusters: list of {'center': (x,y), 'size': n}
        sorted by descending size.
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
                if dist < radius:
                    visited[j] = True
                    cluster_pts.append(j)
                    queue.append(j)

        if len(cluster_pts) >= min_size:
            center = points[cluster_pts].mean(axis=0)
            clusters.append({
                'center': (int(center[0]), int(center[1])),
                'size': len(cluster_pts),
            })

    clusters.sort(key=lambda c: c['size'], reverse=True)
    return clusters
