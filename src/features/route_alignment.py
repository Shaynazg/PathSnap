"""
Reference-route alignment: turns the .route file into ONE labelled arc per GPS
observation (the `is_match` training label).

Why not "candidate arc is anywhere in the route"?  That marks ~40% of candidate
rows positive (loops, revisited/parallel arcs), so an observation has several
"correct" arcs.  Here we solve a monotone alignment instead: observations move
forward along the route, never backwards, minimising total GPS-to-arc distance.
Observations that cannot be aligned get NO label (they are not negatives).
"""
from collections import defaultdict

import pandas as pd


def align_observations(features, route_arc_ids, jump_pen=2.0, free_jump=2,
                       skip_pen=60.0, prune=300.0):
    """
    Parameters
    ----------
    features : DataFrame with obs_idx, candidate_arc_id, distance_m
    route_arc_ids : sequence of arc ids (the .route file, in order)

    Returns
    -------
    dict {obs_idx: arc_id} for aligned observations only.
    """
    occ = defaultdict(list)
    for pos, arc in enumerate(route_arc_ids):
        occ[int(arc)].append(pos)

    groups = {o: g for o, g in features.groupby("obs_idx", sort=True)}
    obs_list = sorted(groups)

    layers, prev = [], {-1: (0.0, None)}          # pos -> (cost, (prev_pos, matched))
    for o in obs_list:
        g = groups[o]
        emission = {}                              # route position -> best distance
        for arc, d in zip(g["candidate_arc_id"], g["distance_m"]):
            for p in occ.get(int(arc), ()):
                if d < emission.get(p, 1e18):
                    emission[p] = d

        cur = {p: (c + skip_pen, (p, False)) for p, (c, _) in prev.items()}  # skip obs
        for p, d in emission.items():
            best = None
            for q, (c, _) in prev.items():
                if q == -1 or p >= q:              # forward-only along the route
                    v = c + d + (0.0 if q == -1 else jump_pen * max(0, p - q - free_jump))
                    if best is None or v < best[0]:
                        best = (v, (q, True))
            if best and best[0] < cur.get(p, (1e18,))[0]:
                cur[p] = best

        floor = min(c for c, _ in cur.values())
        cur = {p: s for p, s in cur.items() if s[0] <= floor + prune}
        layers.append(cur)
        prev = cur

    pos = min(prev, key=lambda k: prev[k][0])
    labels = {}
    for o, cur in zip(reversed(obs_list), reversed(layers)):
        _, (q, matched) = cur[pos]
        if matched:
            labels[o] = int(route_arc_ids[pos])
        pos = q
    return labels


def add_is_match(features, route_arc_ids, **kw):
    """Adds `aligned` (obs has a label) and `is_match` (1/0, only meaningful if aligned)."""
    labels = align_observations(features, route_arc_ids, **kw)
    out = features.copy()
    matched_arc = out["obs_idx"].map(labels)
    out["aligned"] = matched_arc.notna().astype(int)
    out["is_match"] = (out["candidate_arc_id"] == matched_arc).astype(int)
    return out
