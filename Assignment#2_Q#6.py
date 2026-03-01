"""
Dynamic Pathfinding Agent - Informed Search (GBFS & A*)
Grid-based navigation with dynamic obstacles and re-planning.
GUI: Tkinter. No static .txt map files.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import heapq
import time
import random
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass, field


# --- Constants ---
CELL_SIZE = 24
COLOR_EMPTY = "#e8e8e8"
COLOR_WALL = "#2d2d2d"
COLOR_START = "#2196F3"
COLOR_GOAL = "#FF5722"
COLOR_FRONTIER = "#FFEB3B"
COLOR_VISITED = "#9C27B0"
COLOR_PATH = "#4CAF50"
COLOR_AGENT = "#00BCD4"


@dataclass(order=True)
class Node:
    """Node for priority queue: f_score for ordering, (r,c) for identity."""
    f: float
    r: int = field(compare=False)
    c: int = field(compare=False)
    g: float = field(default=0.0, compare=False)
    parent: Optional[Tuple[int, int]] = field(default=None, compare=False)


def manhattan(r1: int, c1: int, r2: int, c2: int) -> float:
    return abs(r1 - r2) + abs(c1 - c2)


def euclidean(r1: int, c1: int, r2: int, c2: int) -> float:
    return ((r1 - r2) ** 2 + (c1 - c2) ** 2) ** 0.5


def get_neighbors(r: int, c: int, rows: int, cols: int) -> List[Tuple[int, int]]:
    out = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def reconstruct_path(parent: dict, goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path


def a_star(
    grid: List[List[int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    heuristic: Callable[[int, int, int, int], float],
) -> Tuple[Optional[List[Tuple[int, int]]], set, set, int]:
    """
    A*: f(n) = g(n) + h(n).
    Returns (path, visited_set, frontier_set, nodes_expanded).
    """
    rows, cols = len(grid), len(grid[0])
    gr, gc = goal
    open_set = [Node(f=0.0, r=start[0], c=start[1], g=0.0)]
    heapq.heapify(open_set)
    g_score = {start: 0.0}
    parent = {}
    visited = set()
    frontier_set = set()
    frontier_set.add(start)
    nodes_expanded = 0

    while open_set:
        node = heapq.heappop(open_set)
        r, c = node.r, node.c
        frontier_set.discard((r, c))
        if (r, c) in visited:
            continue
        visited.add((r, c))
        nodes_expanded += 1

        if (r, c) == goal:
            path = reconstruct_path(parent, goal)
            return path, visited, frontier_set, nodes_expanded

        for nr, nc in get_neighbors(r, c, rows, cols):
            if grid[nr][nc] == 1:
                continue
            if (nr, nc) in visited:
                continue
            tentative_g = g_score[(r, c)] + 1
            if tentative_g < g_score.get((nr, nc), float("inf")):
                g_score[(nr, nc)] = tentative_g
                h = heuristic(nr, nc, gr, gc)
                f = tentative_g + h
                parent[(nr, nc)] = (r, c)
                heapq.heappush(open_set, Node(f=f, r=nr, c=nc, g=tentative_g, parent=(r, c)))
                frontier_set.add((nr, nc))

    return None, visited, frontier_set, nodes_expanded


def gbfs(
    grid: List[List[int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    heuristic: Callable[[int, int, int, int], float],
) -> Tuple[Optional[List[Tuple[int, int]]], set, set, int]:
    """
    Greedy Best-First Search: f(n) = h(n) only.
    Returns (path, visited_set, frontier_set, nodes_expanded).
    """
    rows, cols = len(grid), len(grid[0])
    gr, gc = goal
    h0 = heuristic(start[0], start[1], gr, gc)
    open_set = [Node(f=h0, r=start[0], c=start[1], g=0.0)]
    heapq.heapify(open_set)
    parent = {}
    visited = set()
    frontier_set = set()
    frontier_set.add(start)
    nodes_expanded = 0

    while open_set:
        node = heapq.heappop(open_set)
        r, c = node.r, node.c
        frontier_set.discard((r, c))
        if (r, c) in visited:
            continue
        visited.add((r, c))
        nodes_expanded += 1

        if (r, c) == goal:
            path = reconstruct_path(parent, goal)
            return path, visited, frontier_set, nodes_expanded

        for nr, nc in get_neighbors(r, c, rows, cols):
            if grid[nr][nc] == 1:
                continue
            if (nr, nc) in visited:
                continue
            if (nr, nc) not in parent:
                parent[(nr, nc)] = (r, c)
                h = heuristic(nr, nc, gr, gc)
                heapq.heappush(open_set, Node(f=h, r=nr, c=nc, g=0.0, parent=(r, c)))
                frontier_set.add((nr, nc))

    return None, visited, frontier_set, nodes_expanded


class GridApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Dynamic Pathfinding Agent - GBFS & A*")
        self.rows = 15
        self.cols = 20
        self.grid = [[0] * self.cols for _ in range(self.rows)]
        self.start = (0, 0)
        self.goal = (self.rows - 1, self.cols - 1)
        self.path: Optional[List[Tuple[int, int]]] = None
        self.visited_set: set = set()
        self.frontier_set: set = set()
        self.nodes_expanded = 0
        self.execution_time_ms = 0.0
        self.dynamic_mode = False
        self.agent_index = 0
        self.current_path: Optional[List[Tuple[int, int]]] = None
        self.search_running = False
        self.after_id = None
        self.path_animate_id = None  # animation when Run Search completes (agent moves along path)
        self.place_mode: Optional[str] = None  # "start" or "goal"

        self.heuristics = {
            "Manhattan": manhattan,
            "Euclidean": euclidean,
        }
        self.algorithms = {
            "A*": a_star,
            "GBFS": gbfs,
        }

        self._build_ui()

    def _build_ui(self):
        # Control frame
        ctrl = ttk.Frame(self.root, padding=8)
        ctrl.pack(fill=tk.X)

        ttk.Label(ctrl, text="Rows:").grid(row=0, column=0, padx=2, pady=2)
        self.rows_var = tk.StringVar(value=str(self.rows))
        ttk.Entry(ctrl, textvariable=self.rows_var, width=4).grid(row=0, column=1, padx=2, pady=2)
        ttk.Label(ctrl, text="Cols:").grid(row=0, column=2, padx=2, pady=2)
        self.cols_var = tk.StringVar(value=str(self.cols))
        ttk.Entry(ctrl, textvariable=self.cols_var, width=4).grid(row=0, column=3, padx=2, pady=2)
        ttk.Button(ctrl, text="Resize Grid", command=self._resize).grid(row=0, column=4, padx=4, pady=2)

        ttk.Label(ctrl, text="Obstacle %:").grid(row=1, column=0, padx=2, pady=2)
        self.obstacle_pct_var = tk.StringVar(value="30")
        ttk.Entry(ctrl, textvariable=self.obstacle_pct_var, width=4).grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(ctrl, text="Random Map", command=self._random_map).grid(row=1, column=2, padx=4, pady=2)
        ttk.Button(ctrl, text="Clear Walls", command=self._clear_walls).grid(row=1, column=3, padx=2, pady=2)

        ttk.Label(ctrl, text="Algorithm:").grid(row=2, column=0, padx=2, pady=2)
        self.algo_var = tk.StringVar(value="A*")
        algo_combo = ttk.Combobox(ctrl, textvariable=self.algo_var, values=list(self.algorithms.keys()), state="readonly", width=8)
        algo_combo.grid(row=2, column=1, padx=2, pady=2)
        ttk.Label(ctrl, text="Heuristic:").grid(row=2, column=2, padx=2, pady=2)
        self.heur_var = tk.StringVar(value="Manhattan")
        ttk.Combobox(ctrl, textvariable=self.heur_var, values=list(self.heuristics.keys()), state="readonly", width=10).grid(row=2, column=3, padx=2, pady=2)

        ttk.Button(ctrl, text="Run Search", command=self._run_search).grid(row=3, column=0, columnspan=2, padx=4, pady=4)
        ttk.Button(ctrl, text="Reset View", command=self._reset_view).grid(row=3, column=2, padx=4, pady=4)
        ttk.Button(ctrl, text="Set Start", command=lambda: self._set_place_mode("start")).grid(row=3, column=4, padx=2, pady=4)
        ttk.Button(ctrl, text="Set Goal", command=lambda: self._set_place_mode("goal")).grid(row=3, column=5, padx=2, pady=4)
        self.dynamic_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Dynamic Obstacles", variable=self.dynamic_var, command=self._toggle_dynamic).grid(row=3, column=6, padx=4, pady=4)

        # Metrics
        metrics = ttk.Frame(self.root, padding=8)
        metrics.pack(fill=tk.X)
        self.metrics_label = ttk.Label(metrics, text="Nodes Visited: 0  |  Path Cost: 0  |  Time: 0 ms")
        self.metrics_label.pack(anchor=tk.W)

        # Canvas
        self.canvas = tk.Canvas(
            self.root,
            width=self.cols * CELL_SIZE,
            height=self.rows * CELL_SIZE,
            bg=COLOR_EMPTY,
            highlightthickness=1,
            highlightbackground="#ccc",
        )
        self.canvas.pack(padx=8, pady=8)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Button-3>", self._on_right_click)

        self._draw_grid()

    def _resize(self):
        try:
            r, c = int(self.rows_var.get()), int(self.cols_var.get())
            if r < 2 or c < 2 or r > 50 or c > 60:
                messagebox.showerror("Error", "Rows/Cols between 2–50 and 2–60.")
                return
        except ValueError:
            messagebox.showerror("Error", "Enter valid integers for Rows and Cols.")
            return
        self.rows, self.cols = r, c
        self.grid = [[0] * self.cols for _ in range(self.rows)]
        self.start = (0, 0)
        self.goal = (self.rows - 1, self.cols - 1)
        self.path = None
        self.visited_set = set()
        self.frontier_set = set()
        self.current_path = None
        self.agent_index = 0
        self.canvas.config(width=self.cols * CELL_SIZE, height=self.rows * CELL_SIZE)
        self._draw_grid()

    def _random_map(self):
        try:
            pct = float(self.obstacle_pct_var.get())
            if not (0 <= pct <= 100):
                raise ValueError("Obstacle % must be 0–100")
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
        self.grid = [[0] * self.cols for _ in range(self.rows)]
        n = self.rows * self.cols
        wall_count = int(n * pct / 100)
        cells = [(i // self.cols, i % self.cols) for i in range(n)]
        cells.remove(self.start)
        cells.remove(self.goal)
        random.shuffle(cells)
        for i in range(min(wall_count, len(cells))):
            r, c = cells[i]
            self.grid[r][c] = 1
        self.path = None
        self.visited_set = set()
        self.frontier_set = set()
        self.current_path = None
        self.agent_index = 0
        self._draw_grid()

    def _clear_walls(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[r][c] = 0
        self.path = None
        self.visited_set = set()
        self.frontier_set = set()
        self.current_path = None
        self.agent_index = 0
        self._draw_grid()

    def _set_place_mode(self, mode: str):
        self.place_mode = mode

    def _on_click(self, event):
        if self.search_running:
            return
        c = event.x // CELL_SIZE
        r = event.y // CELL_SIZE
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return
        if self.place_mode == "start":
            if self.grid[r][c] == 0:
                self.start = (r, c)
                self.place_mode = None
                self.path = None
                self.visited_set = set()
                self.frontier_set = set()
                self.current_path = None
            self._draw_grid()
            return
        if self.place_mode == "goal":
            if self.grid[r][c] == 0:
                self.goal = (r, c)
                self.place_mode = None
                self.path = None
                self.visited_set = set()
                self.frontier_set = set()
                self.current_path = None
            self._draw_grid()
            return
        if (r, c) == self.start or (r, c) == self.goal:
            return
        self.grid[r][c] = 1 if self.grid[r][c] == 0 else 0
        self._draw_grid()

    def _on_right_click(self, event):
        if self.search_running:
            return
        c = event.x // CELL_SIZE
        r = event.y // CELL_SIZE
        if not (0 <= r < self.rows and 0 <= c < self.cols) or self.grid[r][c] == 1:
            return
        self.start = (r, c)
        self.place_mode = None
        self.path = None
        self.visited_set = set()
        self.frontier_set = set()
        self.current_path = None
        self._draw_grid()

    def _draw_grid(self):
        self.canvas.delete("all")
        for r in range(self.rows):
            for c in range(self.cols):
                x1, y1 = c * CELL_SIZE, r * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                if self.grid[r][c] == 1:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=COLOR_WALL, outline="#555")
                elif (r, c) == self.start:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=COLOR_START, outline="#1976D2")
                    self.canvas.create_text(x1 + CELL_SIZE // 2, y1 + CELL_SIZE // 2, text="S", font=("Segoe UI", 10, "bold"))
                elif (r, c) == self.goal:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=COLOR_GOAL, outline="#E64A19")
                    self.canvas.create_text(x1 + CELL_SIZE // 2, y1 + CELL_SIZE // 2, text="G", font=("Segoe UI", 10, "bold"))
                elif self.path and (r, c) in self.path:
                    # Green only for path cells the agent has crossed (when animating)
                    if self.current_path and 0 <= self.agent_index < len(self.current_path):
                        crossed = set(self.current_path[: self.agent_index + 1])
                        if (r, c) in crossed:
                            self.canvas.create_rectangle(x1, y1, x2, y2, fill=COLOR_PATH, outline="#388E3C")
                        else:
                            self.canvas.create_rectangle(x1, y1, x2, y2, fill=COLOR_EMPTY, outline="#ccc")
                    else:
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill=COLOR_PATH, outline="#388E3C")
                elif (r, c) in self.frontier_set:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=COLOR_FRONTIER, outline="#F9A825")
                elif (r, c) in self.visited_set:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=COLOR_VISITED, outline="#7B1FA2")
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=COLOR_EMPTY, outline="#ccc")
        # Agent (shown when moving along path after Run Search or in dynamic mode)
        if self.current_path and 0 <= self.agent_index < len(self.current_path):
            ar, ac = self.current_path[self.agent_index]
            ax1, ay1 = ac * CELL_SIZE, ar * CELL_SIZE
            self.canvas.create_oval(ax1 + 2, ay1 + 2, ax1 + CELL_SIZE - 2, ay1 + CELL_SIZE - 2, fill=COLOR_AGENT, outline="#0097A7")

    def _run_search(self):
        if self.grid[self.start[0]][self.start[1]] == 1 or self.grid[self.goal[0]][self.goal[1]] == 1:
            messagebox.showerror("Error", "Start or Goal is blocked.")
            return
        algo_name = self.algo_var.get()
        heur_name = self.heur_var.get()
        algo = self.algorithms.get(algo_name)
        heur = self.heuristics.get(heur_name)
        if not algo or not heur:
            messagebox.showerror("Error", "Select valid Algorithm and Heuristic.")
            return
        self.search_running = True
        t0 = time.perf_counter()
        path, visited, frontier, expanded = algo(self.grid, self.start, self.goal, heur)
        self.execution_time_ms = (time.perf_counter() - t0) * 1000
        self.search_running = False
        self.path = path
        self.visited_set = visited
        self.frontier_set = frontier
        self.nodes_expanded = expanded
        path_cost = len(path) - 1 if path else 0
        self.metrics_label.config(
            text=f"Nodes Visited: {expanded}  |  Path Cost: {path_cost}  |  Time: {self.execution_time_ms:.2f} ms"
        )
        if not path:
            messagebox.showinfo("No Path", "No path found to goal.")
        self.current_path = path
        self.agent_index = 0
        if self.path_animate_id:
            self.root.after_cancel(self.path_animate_id)
            self.path_animate_id = None
        self._draw_grid()
        if path and not self.dynamic_mode:
            self.path_animate_id = self.root.after(200, self._path_animation_step)

    def _reset_view(self):
        if self.path_animate_id:
            self.root.after_cancel(self.path_animate_id)
            self.path_animate_id = None
        self.path = None
        self.visited_set = set()
        self.frontier_set = set()
        self.current_path = None
        self.agent_index = 0
        self.metrics_label.config(text="Nodes Visited: 0  |  Path Cost: 0  |  Time: 0 ms")
        self._draw_grid()

    def _path_animation_step(self):
        """Animate agent along path after Run Search; path turns green as agent crosses it."""
        self.path_animate_id = None
        if self.dynamic_mode or not self.current_path:
            return
        self.agent_index += 1
        self._draw_grid()
        if self.agent_index < len(self.current_path):
            self.path_animate_id = self.root.after(120, self._path_animation_step)

    def _toggle_dynamic(self):
        self.dynamic_mode = self.dynamic_var.get()
        if not self.dynamic_mode:
            if self.after_id:
                self.root.after_cancel(self.after_id)
                self.after_id = None
            return
        if self.path_animate_id:
            self.root.after_cancel(self.path_animate_id)
            self.path_animate_id = None
        if not self.current_path:
            self._run_search()
        if self.current_path:
            self.agent_index = 0
            self.after_id = self.root.after(300, self._dynamic_step)

    def _dynamic_step(self):
        if not self.dynamic_mode or not self.current_path or self.agent_index >= len(self.current_path):
            if self.after_id:
                self.root.after_cancel(self.after_id)
                self.after_id = None
            return
        # Spawn random obstacle with small probability (avoid start/goal/current agent)
        if random.random() < 0.03:
            r, c = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
            if (r, c) != self.start and (r, c) != self.goal:
                self.grid[r][c] = 1
        # Move agent
        cur = self.current_path[self.agent_index]
        if self.grid[cur[0]][cur[1]] == 1:
            # Blocked: re-plan from current position
            start_now = self.current_path[self.agent_index - 1] if self.agent_index > 0 else self.start
            algo = self.algorithms.get(self.algo_var.get())
            heur = self.heuristics.get(self.heur_var.get())
            if algo and heur:
                self.start = start_now
                path, visited, frontier, _ = algo(self.grid, self.start, self.goal, heur)
                if path:
                    self.current_path = path
                    self.agent_index = 0
                    self.visited_set = visited
                    self.frontier_set = frontier
                    self.path = path
                else:
                    self.dynamic_var.set(False)
                    self.dynamic_mode = False
                    messagebox.showinfo("Blocked", "No path after obstacle; dynamic mode off.")
                    self._draw_grid()
                    return
            self._draw_grid()
            self.after_id = self.root.after(150, self._dynamic_step)
            return
        self.agent_index += 1
        if self.agent_index >= len(self.current_path):
            self.dynamic_var.set(False)
            self.dynamic_mode = False
            messagebox.showinfo("Done", "Agent reached goal.")
        self._draw_grid()
        self.after_id = self.root.after(150, self._dynamic_step)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = GridApp()
    app.run()
