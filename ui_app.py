from __future__ import annotations

import json
import math
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from typing import List, Optional

from Graph import Point, build_visibility_graph, dijkstra_path
from random_scane import random_rectangles_scene


"""
Simple scene editor for visibility-graph routing.

Usage options:
1) Load from JSON file (button “Зчитати з файлу”). Format:
   {
     "start": [x, y],
     "goal": [x, y],
     "polygons": [
       [[x1, y1], [x2, y2], [x3, y3]],
       ...
     ]
   }
   Polygons are taken as-is (no automatic closing).
2) Manual placement:
   - Select “Початок” then click canvas to place S.
   - Select “Кінець” then click canvas to place T.
   - Select “Точки перешкоди”, click vertices in order.
   - Press “Замкнути перешкоду” to add the polygon. Repeat for more.
Press “Знайти шлях” to build the visibility graph and run Dijkstra.
"""


RADIUS = 4


class SceneEditor:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Visibility graph editor")

        self.polygons: List[List[Point]] = []
        self.current_poly: List[Point] = []
        self.start: Optional[Point] = None
        self.goal: Optional[Point] = None
        self.path: List[Point] = []
        self.path_length: Optional[float] = None

        self.mode = tk.StringVar(value="start")

        self.canvas = tk.Canvas(root, width=900, height=650, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        controls = tk.Frame(root)
        controls.grid(row=0, column=1, sticky="ns", padx=10, pady=10)

        tk.Label(controls, text="Режим").pack(anchor="w")
        tk.Radiobutton(controls, text="Початок (S)", variable=self.mode, value="start").pack(anchor="w")
        tk.Radiobutton(controls, text="Кінець (T)", variable=self.mode, value="goal").pack(anchor="w")
        tk.Radiobutton(controls, text="Точки перешкоди", variable=self.mode, value="polygon").pack(anchor="w")

        tk.Button(controls, text="Замкнути перешкоду", command=self.finish_polygon).pack(fill="x", pady=(8, 2))
        tk.Button(controls, text="Скинути поточну", command=self.clear_current_polygon).pack(fill="x")

        tk.Label(controls, text="Файл / сцена").pack(anchor="w", pady=(12, 0))
        tk.Button(controls, text="Зчитати з файлу", command=self.load_from_file).pack(fill="x")
        tk.Button(controls, text="Згенерувати випадково", command=self.generate_random_scene).pack(fill="x", pady=(2, 2))
        tk.Button(controls, text="Очистити все", command=self.clear_scene).pack(fill="x", pady=(0, 8))

        tk.Button(controls, text="Знайти шлях", command=self.solve).pack(fill="x")
        self.path_label = tk.Label(controls, text="Довжина шляху: -")
        self.path_label.pack(anchor="w", pady=(6, 0))

        hint = (
            "1) Оберіть режим і клацніть на полотні.\n"
            "2) Для перешкоди натисніть точки у порядку,\n"
            "   потім «Замкнути перешкоду».\n"
            "3) Або зчитайте сцену з JSON-файлу."
        )
        tk.Label(controls, text=hint, justify="left").pack(anchor="w", pady=(12, 0))

    # ---------- Canvas helpers ----------

    def redraw(self) -> None:
        self.canvas.delete("all")

        # Polygons
        for poly in self.polygons:
            if len(poly) >= 2:
                coords = [c for p in poly for c in (p.x, p.y)]
                self.canvas.create_polygon(*coords, outline="#777", fill="#ddd", width=2)

        # Current polygon (not yet closed)
        if self.current_poly:
            coords = [c for p in self.current_poly for c in (p.x, p.y)]
            self.canvas.create_line(*coords, fill="#999", dash=(4, 2), width=2)
            for p in self.current_poly:
                self.canvas.create_oval(p.x - RADIUS, p.y - RADIUS, p.x + RADIUS, p.y + RADIUS, fill="#999", outline="")

        # Path
        if self.path and len(self.path) >= 2:
            coords = [c for p in self.path for c in (p.x, p.y)]
            self.canvas.create_line(*coords, fill="#1a73e8", width=3)

        # Start/goal markers
        if self.start:
            self.canvas.create_oval(
                self.start.x - RADIUS, self.start.y - RADIUS,
                self.start.x + RADIUS, self.start.y + RADIUS,
                fill="#34a853", outline=""
            )
            self.canvas.create_text(self.start.x, self.start.y - 10, text="S", fill="#34a853")
        if self.goal:
            self.canvas.create_oval(
                self.goal.x - RADIUS, self.goal.y - RADIUS,
                self.goal.x + RADIUS, self.goal.y + RADIUS,
                fill="#ea4335", outline=""
            )
            self.canvas.create_text(self.goal.x, self.goal.y - 10, text="T", fill="#ea4335")

    # ---------- Actions ----------

    def on_canvas_click(self, event: tk.Event) -> None:
        p = Point(float(event.x), float(event.y))

        if self.mode.get() == "start":
            self.start = p
        elif self.mode.get() == "goal":
            self.goal = p
        elif self.mode.get() == "polygon":
            self.current_poly.append(p)
        else:
            messagebox.showinfo("Режим не обрано", "Оберіть, що ви ставите: S, T чи вершини перешкоди.")
            return

        self.path = []
        self.path_length = None
        self.update_path_label()
        self.redraw()

    def finish_polygon(self) -> None:
        if len(self.current_poly) < 3:
            messagebox.showwarning("Мало точок", "Полігон має містити щонайменше 3 вершини.")
            return
        self.polygons.append(self.current_poly.copy())
        self.current_poly.clear()
        self.path = []
        self.path_length = None
        self.update_path_label()
        self.redraw()

    def clear_current_polygon(self) -> None:
        self.current_poly.clear()
        self.redraw()

    def clear_scene(self) -> None:
        self.polygons.clear()
        self.current_poly.clear()
        self.start = None
        self.goal = None
        self.path = []
        self.path_length = None
        self.update_path_label()
        self.redraw()

    def load_from_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Виберіть JSON файл сцени",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Помилка читання", f"Не вдалося прочитати файл:\n{exc}")
            return

        try:
            self.start = Point(*data["start"])
            self.goal = Point(*data["goal"])
            self.polygons = [
                [Point(*v) for v in poly]
                for poly in data.get("polygons", [])
            ]
            self.current_poly.clear()
            self.path = []
            self.path_length = None
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Невірний формат", f"Очікувано ключі start, goal, polygons.\n{exc}")
            return

        self.update_path_label()
        self.redraw()

    def solve(self) -> None:
        if self.current_poly:
            if not messagebox.askyesno(
                "Незавершена перешкода",
                "Є незамкнений полігон. Замкнути його і продовжити?"
            ):
                return
            if len(self.current_poly) < 3:
                messagebox.showwarning("Мало точок", "Полігон має містити щонайменше 3 вершини.")
                return
            self.polygons.append(self.current_poly.copy())
            self.current_poly.clear()

        if not self.start or not self.goal:
            messagebox.showwarning("Немає S або T", "Поставте точки S та T.")
            return

        try:
            adj, _ = build_visibility_graph(self.polygons, self.start, self.goal)
            length, path = dijkstra_path(adj, self.start, self.goal)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Помилка розрахунку", str(exc))
            return

        if math.isinf(length):
            messagebox.showinfo("Шлях не знайдено", "S і T не з'єднані.")
            self.path = []
            self.path_length = None
        else:
            self.path = path
            self.path_length = length

        self.update_path_label()
        self.redraw()

    def update_path_label(self) -> None:
        if self.path_length is None:
            self.path_label.config(text="Довжина шляху: -")
        else:
            self.path_label.config(text=f"Довжина шляху: {self.path_length:.2f}")

    def generate_random_scene(self) -> None:
        n_obs = simpledialog.askinteger("Кількість перешкод", "Скільки перешкод згенерувати?", initialvalue=6, minvalue=0)
        if n_obs is None:
            return

        width = int(self.canvas.cget("width"))
        height = int(self.canvas.cget("height"))

        try:
            polygons, start, goal = random_rectangles_scene(
                Point,
                n_obs=n_obs,
                bbox=(0.0, 0.0, float(width), float(height)),
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Помилка генерації", str(exc))
            return

        self.polygons = polygons
        self.current_poly.clear()
        self.start = start
        self.goal = goal
        self.path = []
        self.path_length = None
        self.update_path_label()
        self.redraw()


def main() -> None:
    root = tk.Tk()
    editor = SceneEditor(root)
    editor.redraw()
    root.mainloop()


if __name__ == "__main__":
    main()
