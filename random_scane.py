
# random_scene.py
from __future__ import annotations

from typing import Optional, Any
import random
import math


def _random_poly_in_rect(PointCls,
                         x0: float, y0: float,
                         x1: float, y1: float,
                         min_vertices: int = 3,
                         max_vertices: int = 8,
                         min_r: float = 0.8,   # ← додаємо параметр
                         max_r: float = 1.0):
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    rx = 0.5 * (x1 - x0)
    ry = 0.5 * (y1 - y0)

    k = random.randint(min_vertices, max_vertices)
    angles = sorted(random.uniform(0.0, 2.0 * math.pi) for _ in range(k))

    pts = []
    for a in angles:
        r = random.uniform(min_r, max_r)   # було (0.4, 1.0)
        px = cx + r * rx * math.cos(a)
        py = cy + r * ry * math.sin(a)
        pts.append(PointCls(px, py))

    return pts

def _aabb_overlap(a, b, margin: float = 0.0) -> bool:
    # a,b: (x0,y0,x1,y1), with x0<x1, y0<y1
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ax0 -= margin; ay0 -= margin; ax1 += margin; ay1 += margin
    bx0 -= margin; by0 -= margin; bx1 += margin; by1 += margin
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def _point_in_rect(p, r, eps=1e-12) -> bool:
    x0, y0, x1, y1 = r
    return (x0 + eps <= p.x <= x1 - eps) and (y0 + eps <= p.y <= y1 - eps)


def _sample_free_point(PointCls, bbox, rects, tries=10_000):
    xmin, ymin, xmax, ymax = bbox
    for _ in range(tries):
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        p = PointCls(x, y)
        ok = True
        for r in rects:
            if _point_in_rect(p, r):
                ok = False
                break
        if ok:
            return p
    raise RuntimeError("Не вдалося знайти точку поза перешкодами. Зменш перешкоди або їх кількість.")


def random_rectangles_scene(PointCls,
                            n_obs: int = 8,
                            bbox: tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0),
                            size_range: tuple[float, float] = (8.0, 22.0),
                            margin: float = 2.0,
                            seed: Optional[int] = None,
                            start_goal_min_dist: float = 50.0,
                            max_tries: int = 50_000):
    """
    Генерує сцену з неперетинними прямокутниками.
    Повертає: (polygons, start, goal)

    - PointCls: твій клас Point (конструктор Point(x,y))
    - bbox: (xmin, ymin, xmax, ymax)
    - margin: “зазор” між прямокутниками
    """
    if seed is not None:
        random.seed(seed)

    xmin, ymin, xmax, ymax = bbox
    smin, smax = size_range

    rects: list[tuple[float, float, float, float]] = []

    tries = 0
    while len(rects) < n_obs and tries < max_tries:
        tries += 1

        w = random.uniform(smin, smax)
        h = random.uniform(smin, smax)

        x0 = random.uniform(xmin, xmax - w)
        y0 = random.uniform(ymin, ymax - h)
        x1 = x0 + w
        y1 = y0 + h

        cand = (x0, y0, x1, y1)

        ok = True
        for r in rects:
            if _aabb_overlap(cand, r, margin=margin):
                ok = False
                break

        if ok:
            rects.append(cand)

    if len(rects) < n_obs:
        raise RuntimeError(f"Не вдалося згенерувати {n_obs} неперетинних прямокутників. "
                           f"Спробуй менше n_obs або менші size_range, або більший bbox.")

    polygons = [_random_poly_in_rect(PointCls, *r) for r in rects]

    # Старт і фініш (не всередині перешкод) + хоч якась відстань між ними
    for _ in range(2000):
        start = _sample_free_point(PointCls, bbox, rects)
        goal = _sample_free_point(PointCls, bbox, rects)
        dx = start.x - goal.x
        dy = start.y - goal.y
        if math.hypot(dx, dy) >= start_goal_min_dist:
            return polygons, start, goal

    # якщо не вдалось набрати min_dist — повернемо як є
    return polygons, start, goal