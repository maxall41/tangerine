import pickle
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import split
from tqdm import tqdm


def make_ring_polygons(outer, inner):
    """Split outer polygon into two non-intersecting polygons forming a ring."""
    outer_poly = Polygon(outer)
    inner_poly = Polygon(inner)

    # Compute centroids
    c_outer = np.array(outer_poly.centroid.coords[0])
    c_inner = np.array(inner_poly.centroid.coords[0])
    direction = c_inner - c_outer
    if np.allclose(direction, 0):
        direction = np.array([1, 0])
    direction = direction / np.linalg.norm(direction)

    # Extend line beyond bounds
    minx, miny, maxx, maxy = outer_poly.bounds
    diag = np.hypot(maxx - minx, maxy - miny)
    line_start = c_outer - direction * diag * 2
    line_end = c_outer + direction * diag * 2
    split_line = LineString([tuple(line_start), tuple(line_end)])

    # Split outer polygon
    pieces = split(outer_poly, split_line)
    ring_segments = []

    for piece in pieces.geoms:
        # If piece overlaps inner polygon, cut out overlap
        if piece.intersects(inner_poly):
            piece = piece.difference(inner_poly)
        ring_segments.append(piece)

    # Convert to simple list-of-lists format
    result = []
    for p in ring_segments:
        if p.geom_type == "Polygon":
            result.append(np.array([list(pt) for pt in p.exterior.coords]))
        elif p.geom_type == "MultiPolygon":
            for sub in p.geoms:
                result.append(np.array([list(pt) for pt in sub.exterior.coords]))
    return result


def process_ring_polygon(args):
    """Worker function for parallel ring structure processing."""
    i, outer, shapely_polys, used = args
    if i in used:
        return None

    contained = None
    contained_idx = None
    for j, inner in enumerate(shapely_polys):
        if i == j or j in used:
            continue
        if outer.contains(inner):
            contained = inner
            contained_idx = j
            break

    if contained:
        ring_parts = make_ring_polygons(list(outer.exterior.coords), list(contained.exterior.coords))
        return (i, contained_idx, ring_parts)
    return (i, None, [np.array([list(pt) for pt in outer.exterior.coords])])


def polygons_to_ring_structure(polygons):
    shapely_polys = [Polygon(p) for p in polygons]
    used = set()
    results = []

    for i, outer in tqdm(enumerate(shapely_polys), desc="Extracting ring structures from holes..."):
        if i in used:
            continue
        contained = None
        for j, inner in enumerate(shapely_polys):
            if i == j or j in used:
                continue
            if outer.contains(inner):
                contained = inner
                used.add(j)
                break

        if contained:
            ring_parts = make_ring_polygons(list(outer.exterior.coords), list(contained.exterior.coords))
            results.extend(ring_parts)
        else:
            results.append(np.array([list(pt) for pt in outer.exterior.coords]))

        used.add(i)
    return results


def process_single_polygon(args):
    """Worker function for parallel polygon splitting."""
    poly_coords, max_chord_distance, min_contour_distance, chord_contour_ratio = args

    # Create shapely polygon
    poly = Polygon(poly_coords)

    if not poly.is_valid:
        poly = poly.buffer(0)

    # Find bridge using chord analysis
    bridge_cuts = find_bridges_chord_method(
        poly_coords,
        max_chord_distance,
        min_contour_distance,
        chord_contour_ratio,
    )

    if not bridge_cuts:
        return [poly_coords]

    # Split polygon at bridges
    split_polys = split_polygon_at_bridges(poly, bridge_cuts)

    # Convert back to numpy arrays
    result = []
    if len(split_polys) > 1:
        for sp in split_polys:
            if isinstance(sp, Polygon) and not sp.is_empty:
                result.append(np.array(sp.exterior.coords[:-1]))
    else:
        result.append(poly_coords)

    return result


def split_bridged_polygons(
    polygons: list[np.ndarray],
    max_chord_distance: float = 100.0,
    min_contour_distance: int = 20,
    chord_contour_ratio: float = 0.15,
    n_jobs: int = None,
) -> list[np.ndarray]:
    """Split polygons connected by bridges using chord analysis.

    Skips polygons that are fully contained within any other polygon (they are
    returned unchanged).

    Parameters
    ----------
    polygons : List[np.ndarray]
        List of polygons, each as Nx2 numpy array of [x, y] coordinates
    max_chord_distance : float
        Maximum straight-line distance between points to consider as potential bridge
    min_contour_distance : int
        Minimum number of points along contour between the pair
    chord_contour_ratio : float
        Maximum ratio of (chord_distance / contour_path_length) to be a bridge
    n_jobs : int
        Number of parallel jobs. If None, uses cpu_count()

    Returns
    -------
    List[np.ndarray]
        List of split polygons (order preserved)

    """
    if n_jobs is None:
        n_jobs = cpu_count()

    # Build shapely polygons once for containment checks
    shapely_polys = [Polygon(p) for p in polygons]

    # Determine indices that are fully contained in another polygon and should be skipped
    skip_indices = set()
    n = len(shapely_polys)
    for i in range(n):
        pi = shapely_polys[i]
        # If pi is empty/invalid, skip containment check (treat as non-contained)
        if pi.is_empty:
            continue
        for j in range(n):
            if i == j:
                continue
            pj = shapely_polys[j]
            if pj.is_empty:
                continue
            # If pj strictly contains pi, then skip processing i
            if pj.contains(pi):
                skip_indices.add(i)
                break

    # Prepare tasks for only the polygons that are NOT skipped, in original order
    task_indices = [i for i in range(len(polygons)) if i not in skip_indices]
    tasks = [(polygons[i], max_chord_distance, min_contour_distance, chord_contour_ratio) for i in task_indices]

    result_polygons: list[np.ndarray] = []

    # Use a pool to process only the non-contained polygons; imap preserves task order
    if len(tasks) > 0:
        with Pool(n_jobs) as pool:
            task_iter = pool.imap(process_single_polygon, tasks)
            # Reconstruct final list preserving original order:
            # iterate through original polygon indices and either append the skipped polygon
            # unchanged or the next result from the pool.
            task_iter = iter(task_iter)
            for i in range(len(polygons)):
                if i in skip_indices:
                    # Append original polygon as-is (no splitting performed)
                    result_polygons.append(polygons[i])
                else:
                    split_results = next(task_iter)
                    # split_results is a list of numpy arrays (one or more parts)
                    result_polygons.extend(split_results)
    else:
        # No tasks (all polygons skipped) — just return original polygons in order
        result_polygons = list(polygons)

    return result_polygons


def find_bridges_chord_method(
    coords: np.ndarray,
    max_chord_distance: float,
    min_contour_distance: int,
    chord_contour_ratio: float,
) -> list[tuple]:
    """Fast bridge detection using precomputed pairwise distances and cumulative path lengths."""
    n = len(coords)
    if n < min_contour_distance * 2:
        return []

    # Precompute pairwise Euclidean distances (O(n²) but vectorized)
    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.hypot(diffs[..., 0], diffs[..., 1])

    # Precompute cumulative contour distances
    segment_lengths = np.hypot(
        np.diff(coords[:, 0], append=coords[0, 0]),
        np.diff(coords[:, 1], append=coords[0, 1]),
    )
    cumlen = np.cumsum(segment_lengths)
    total_len = cumlen[-1]

    bridge_candidates = []
    sample_stride = max(1, n // 200)

    for i in range(0, n, sample_stride):
        # vectorized selection of j indices far enough apart
        js = np.arange(i + min_contour_distance, n - min_contour_distance, sample_stride)
        if len(js) == 0:
            continue

        chord_dists = dists[i, js]
        valid_mask = chord_dists <= max_chord_distance
        if not np.any(valid_mask):
            continue

        js = js[valid_mask]
        chord_dists = chord_dists[valid_mask]

        # contour path distances using cumulative lengths (both directions)
        path_fw = np.abs(cumlen[js % n] - cumlen[i])
        path_fw = np.minimum(path_fw, total_len - path_fw)
        path_fw = np.maximum(path_fw, 1e-6)

        ratios = chord_dists / path_fw
        valid = ratios < chord_contour_ratio
        if np.any(valid):
            for jj, ratio, cd, pd in zip(js[valid], ratios[valid], chord_dists[valid], path_fw[valid], strict=False):
                bridge_candidates.append(
                    {
                        "p1": coords[i],
                        "p2": coords[jj],
                        "idx1": i,
                        "idx2": jj,
                        "ratio": ratio,
                        "chord": cd,
                        "path": pd,
                    },
                )

    if not bridge_candidates:
        return []

    # Sort and select best non-overlapping bridges
    bridge_candidates.sort(key=lambda x: x["ratio"])
    selected = []
    for cand in bridge_candidates[:5]:
        i1, i2 = cand["idx1"], cand["idx2"]
        if any(
            abs(i1 - s["idx1"]) < min_contour_distance
            or abs(i2 - s["idx2"]) < min_contour_distance
            or abs(i1 - s["idx2"]) < min_contour_distance
            or abs(i2 - s["idx1"]) < min_contour_distance
            for s in selected
        ):
            continue
        selected.append(cand)

    return [(b["p1"], b["p2"]) for b in selected]


def calculate_path_length(coords: np.ndarray, idx1: int, idx2: int, forward: bool) -> float:
    """Calculate actual path length along contour between two indices."""
    n = len(coords)
    length = 0.0

    if forward:
        i = idx1
        while i != idx2:
            next_i = (i + 1) % n
            length += np.linalg.norm(coords[next_i] - coords[i])
            i = next_i
            if i == idx1:  # Safety check
                break
    else:
        i = idx1
        while i != idx2:
            prev_i = (i - 1) % n
            length += np.linalg.norm(coords[prev_i] - coords[i])
            i = prev_i
            if i == idx1:  # Safety check
                break

    return length


def split_polygon_at_bridges(poly: Polygon, bridge_cuts: list[tuple]) -> list[Polygon]:
    """Split polygon at detected bridge cut lines."""
    result = [poly]

    for p1, p2 in bridge_cuts:
        new_result = []

        for current_poly in result:
            # Create cutting line with extension
            direction = np.array(p2) - np.array(p1)
            length = np.linalg.norm(direction)

            if length > 0:
                direction = direction / length
                extension = length * 0.5
                extended_p1 = p1 - direction * extension
                extended_p2 = p2 + direction * extension
                cut_line = LineString([extended_p1, extended_p2])
            else:
                cut_line = LineString([p1, p2])

            try:
                split_result = split(current_poly, cut_line)

                if hasattr(split_result, "geoms") and len(split_result.geoms) > 1:
                    for geom in split_result.geoms:
                        if isinstance(geom, Polygon) and geom.area > 1e-6:
                            new_result.append(geom)
                else:
                    new_result.append(current_poly)
            except Exception:
                new_result.append(current_poly)

        result = new_result

    return result


def outlines_list(masks: np.ndarray) -> list[np.ndarray]:
    """Get outlines of masks as a list of ROIs, including holes as separate polygons."""
    outpix = []
    for n in np.unique(masks)[1:]:
        mn = masks == n
        if mn.sum() > 0:
            # Use RETR_CCOMP to get both external contours and holes
            contours, hierarchy = cv2.findContours(
                mn.astype(np.uint8),
                mode=cv2.RETR_CCOMP,
                method=cv2.CHAIN_APPROX_NONE,
            )

            if hierarchy is None or len(contours) == 0:
                outpix.append(np.zeros((0, 2)))
                continue

            # hierarchy format: [Next, Previous, First_Child, Parent]
            hierarchy = hierarchy[0]

            # Add all contours (both external and holes)
            for i, contour in enumerate(contours):
                pix = contour.astype(int).squeeze()

                # Handle single-point contours
                if pix.ndim == 1:
                    pix = pix.reshape(1, -1)

                if len(pix) > 4:
                    outpix.append(pix)
                else:
                    outpix.append(np.zeros((0, 2)))

    with open("polygons.pkl", "wb") as f:
        pickle.dump(outpix, f)
    return polygons_to_ring_structure(
        split_bridged_polygons(
            outpix,
            max_chord_distance=80,
            min_contour_distance=15,
            chord_contour_ratio=0.2,
        ),
    )


def polygon_area(pts):
    """Calculates the area of a polygon using the Shoelace formula.
    'pts' is expected to be a numpy array of shape (N, 2).
    """
    x = pts[:, 0]
    y = pts[:, 1]
    # The formula is 0.5 * |(x1*y2 + x2*y3 + ... + xn*y1) - (y1*x2 + y2*x3 + ... + yn*x1)|
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
