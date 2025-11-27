# Graph.py
from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
from typing import List, Dict, Tuple, Optional
import math
import heapq

from fontTools.ttLib.woff2 import bboxFormat
from sortedcontainers import SortedList  # pip install sortedcontainers

EPS = 1e-9

CG_COLLINEAR = 0
CG_LEFT = 1
CG_RIGHT = -1


# ===================== Primitives =====================

@dataclass(frozen=True, order=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Segment:
    a: Point
    b: Point

    def __getitem__(self, i: int) -> Point:
        return self.a if i == 0 else self.b


# ===================== Geometry =====================

def dist(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def orientation(a: Point, b: Point, c: Point) -> int:
    cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    if cross > EPS:
        return CG_LEFT
    if cross < -EPS:
        return CG_RIGHT
    return CG_COLLINEAR


def collinear_are_ordered_along_line(v: Point, a: Point, b: Point) -> bool:
    da = (a.x - v.x) ** 2 + (a.y - v.y) ** 2
    db = (b.x - v.x) ** 2 + (b.y - v.y) ** 2
    if abs(da - db) > EPS:
        return da < db
    return (a.x, a.y) < (b.x, b.y)


def on_segment(a: Point, b: Point, c: Point) -> bool:
    return (min(a.x, b.x) - EPS <= c.x <= max(a.x, b.x) + EPS and
            min(a.y, b.y) - EPS <= c.y <= max(a.y, b.y) + EPS)


def has_intersection(s1: Segment, s2: Segment) -> bool:
    p1, p2 = s1.a, s1.b
    q1, q2 = s2.a, s2.b

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    if o1 == CG_COLLINEAR and on_segment(p1, p2, q1):
        return True
    if o2 == CG_COLLINEAR and on_segment(p1, p2, q2):
        return True
    if o3 == CG_COLLINEAR and on_segment(q1, q2, p1):
        return True
    if o4 == CG_COLLINEAR and on_segment(q1, q2, p2):
        return True

    return False


# ===================== Vertical-ray init helper =====================

def orient_segment_wrt_v(v: Point, seg: Segment) -> Segment:
    a, b = seg.a, seg.b
    ori = orientation(v, a, b)
    if ori == CG_LEFT or (ori == CG_COLLINEAR and collinear_are_ordered_along_line(v, b, a)):
        a, b = b, a
    return Segment(a, b)


def check_if_segment_intersects_vertical_ray_from_v(v: Point, seg_oriented: Segment) -> bool:
    if v == seg_oriented.a or v == seg_oriented.b:
        return False

    vdown = Point(v.x, v.y - 1)
    or1 = orientation(vdown, v, seg_oriented.a)
    or2 = orientation(vdown, v, seg_oriented.b)

    return ((or1 == CG_LEFT or (or1 == CG_COLLINEAR and seg_oriented.a.y >= v.y)) and
            (or2 == CG_RIGHT or (or2 == CG_COLLINEAR and seg_oriented.b.y >= v.y)))


# ===================== Visibility graph O(n^2 log n) =====================

def _normalize_contour(contour: List[Point]) -> List[Point]:
    if len(contour) >= 2 and contour[0] == contour[-1]:
        return contour[:-1]
    return contour


def visibility_graph_segments(
    contours: List[List[Point]],
    extra_points: Optional[List[Point]] = None
) -> Tuple[List[Segment], List[Segment], List[Point], Dict[Point, Tuple[int, int]], Dict[int, int]]:
    """
    Returns:
      edges_vis        : visibility edges (directed as in original sweep; we will symmetrize later)
      edges_obstacles  : obstacle boundary edges
      all_points       : all points (vertices + extra_points)
      owner            : map Point -> (poly_id, idx_in_poly) for obstacle vertices only
      poly_len         : map poly_id -> number of vertices
    """
    extra_points = extra_points or []

    points_set = set(extra_points)

    # obstacle boundary edges
    edges_obstacles: List[Segment] = []

    # ownership to forbid non-adjacent edges inside same rectangle/polygon
    owner: Dict[Point, Tuple[int, int]] = {}
    poly_len: Dict[int, int] = {}

    for pid, contour in enumerate(contours):
        poly = _normalize_contour(contour)
        m = len(poly)
        if m < 2:
            continue

        poly_len[pid] = m

        for i, p in enumerate(poly):
            points_set.add(p)
            owner[p] = (pid, i)

        for i in range(m):
            a = poly[i]
            b = poly[(i + 1) % m]
            if a != b:
                edges_obstacles.append(Segment(a, b))

    points = sorted(points_set)

    edges_vis: List[Segment] = []

    def are_adjacent(pid: int, i: int, j: int) -> bool:
        m = poly_len[pid]
        d = abs(i - j)
        return d == 1 or d == m - 1

    def allow_visibility_edge(v: Point, p: Point) -> bool:
        ov = owner.get(v)
        op = owner.get(p)
        if ov is None or op is None:
            return True  # start/goal (or any extra point)
        if ov[0] != op[0]:
            return True  # different polygons
        pid = ov[0]
        return are_adjacent(pid, ov[1], op[1])  # forbid diagonals inside same polygon

    # sweep for each vertex
    for v in points:
        cur_points = [q for q in points if q.x >= v.x and q != v]

        def angle_lt(a: Point, b: Point) -> bool:
            ori = orientation(v, a, b)
            if ori == CG_COLLINEAR:
                return collinear_are_ordered_along_line(v, a, b)
            return ori == CG_RIGHT

        def angle_cmp(a: Point, b: Point) -> int:
            if a == b:
                return 0
            ab = angle_lt(a, b)
            ba = angle_lt(b, a)
            if ab and not ba:
                return -1
            if ba and not ab:
                return 1
            return 0

        cur_points.sort(key=cmp_to_key(angle_cmp))

        oriented_segments: List[Segment] = []
        begins: Dict[Point, List[int]] = {}
        ends: Dict[Point, List[int]] = {}

        for i, seg in enumerate(edges_obstacles):
            seg_o = orient_segment_wrt_v(v, seg)
            oriented_segments.append(seg_o)
            begins.setdefault(seg_o.a, []).append(i)
            ends.setdefault(seg_o.b, []).append(i)

        def seg_lt(idx_a: int, idx_b: int) -> bool:
            if idx_a == idx_b:
                return False

            a = oriented_segments[idx_a]
            b = oriented_segments[idx_b]

            if a.a == b.a:
                return orientation(a.b, b.a, b.b) == CG_RIGHT

            if a.b == b.b:
                return orientation(b.a, b.b, a.a) == CG_RIGHT

            ori = orientation(v, a.a, b.a)

            if ori == CG_COLLINEAR:
                return collinear_are_ordered_along_line(v, a.a, b.a)

            return ((ori == CG_RIGHT and orientation(a.b, a.a, b.a) == CG_RIGHT) or
                    (ori == CG_LEFT and orientation(b.a, b.b, a.a) == CG_RIGHT))

        class SegRef:
            __slots__ = ("idx",)

            def __init__(self, idx: int):
                self.idx = idx

            def __eq__(self, other: object) -> bool:
                return isinstance(other, SegRef) and self.idx == other.idx

            def __lt__(self, other: "SegRef") -> bool:
                ab = seg_lt(self.idx, other.idx)
                if ab:
                    return True
                ba = seg_lt(other.idx, self.idx)
                if ba:
                    return False
                return self.idx < other.idx

            def __hash__(self) -> int:
                return hash(self.idx)

        status = SortedList()
        active = set()

        # initialize status with segments intersecting vertical down-ray from v
        for i in range(len(oriented_segments)):
            if check_if_segment_intersects_vertical_ray_from_v(v, oriented_segments[i]):
                status.add(SegRef(i))
                active.add(i)

        for p in cur_points:
            if len(status) > 0:
                closest = oriented_segments[status[0].idx]
                if (not has_intersection(Segment(v, p), closest) or closest.a == p or closest.b == p):
                    if allow_visibility_edge(v, p):
                        edges_vis.append(Segment(v, p))
            else:
                if allow_visibility_edge(v, p):
                    edges_vis.append(Segment(v, p))

            # update status: erase segments ending at p
            for i in ends.get(p, []):
                if i in active:
                    status.remove(SegRef(i))
                    active.remove(i)

            # insert segments starting at p
            for i in begins.get(p, []):
                if i not in active:
                    status.add(SegRef(i))
                    active.add(i)

    return edges_vis, edges_obstacles, points, owner, poly_len


def build_visibility_graph(
    contours: List[List[Point]],
    start: Point,
    goal: Point
) -> Tuple[Dict[Point, List[Tuple[Point, float]]], List[Point]]:
    edges_vis, edges_obs, points, owner, poly_len = visibility_graph_segments(
        contours, extra_points=[start, goal]
    )

    adj: Dict[Point, List[Tuple[Point, float]]] = {p: [] for p in points}

    # visibility edges undirected
    for e in edges_vis:
        w = dist(e.a, e.b)
        adj[e.a].append((e.b, w))
        adj[e.b].append((e.a, w))

    # obstacle boundary edges undirected (allow walking along walls)
    for e in edges_obs:
        w = dist(e.a, e.b)
        adj[e.a].append((e.b, w))
        adj[e.b].append((e.a, w))

    return adj, points


# ===================== Dijkstra =====================

def dijkstra_path(
    adj: Dict[Point, List[Tuple[Point, float]]],
    start: Point,
    goal: Point
) -> Tuple[float, List[Point]]:
    INF = float("inf")
    dist_map: Dict[Point, float] = {start: 0.0}
    prev: Dict[Point, Point] = {}
    pq: List[Tuple[float, Point]] = [(0.0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist_map.get(u, INF):
            continue
        if u == goal:
            break

        for v, w in adj.get(u, []):
            nd = d + w
            if nd + 1e-15 < dist_map.get(v, INF):
                dist_map[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if goal not in dist_map:
        return INF, []

    path = [goal]
    cur = goal
    while cur != start:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return dist_map[goal], path


# ===================== Plot (optional) =====================

def plot_scene(polygons: List[List[Point]], start: Point, goal: Point, path: List[Point]):
    import matplotlib.pyplot as plt

    def close(poly: List[Point]) -> List[Point]:
        poly = _normalize_contour(poly)
        return poly + [poly[0]] if poly else poly

    for poly in polygons:
        c = close(poly)
        xs = [p.x for p in c]
        ys = [p.y for p in c]
        plt.plot(xs, ys)

    plt.scatter([start.x], [start.y], marker="o")
    plt.scatter([goal.x], [goal.y], marker="x")

    if path:
        px = [p.x for p in path]
        py = [p.y for p in path]
        plt.plot(px, py)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


# ===================== Demo with your generator =====================

def demo_with_random_scene(seed: int = None, n_obs: int = 8, bbox: tuple[float, float, float, float] = (0, 0, 100, 100)):
    from random_scane import random_rectangles_scene  # your file random_scene.py must be рядом

    polygons, start, goal = random_rectangles_scene(Point, n_obs=n_obs, seed=seed, bbox=bbox)
    adj, _ = build_visibility_graph(polygons, start, goal)
    length, path = dijkstra_path(adj, start, goal)

    print("start =", start, "goal =", goal)
    print("path length =", length)
    print("path vertices =", path)

    plot_scene(polygons, start, goal, path)


if __name__ == "__main__":
    demo_with_random_scene(seed=None, n_obs=20, bbox= (0, 0, 100, 100))
