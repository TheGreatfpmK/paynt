"""
Extreme points of the transition polytopes of an interval POMDP (PRISM file, stormpy >= 2.0).

For every choice (s, a) with successors t_1..t_k and intervals [l_i, u_i], computes the
vertices of  { p : l <= p <= u, sum(p) = 1 }.
"""

import json
import os
import re
import tarfile
import tempfile
from dataclasses import dataclass

import numpy as np
import stormpy

def find_extreme_points(lower, upper, tol=1e-12):
    """
    Find the vertices of {p : lower <= p <= upper, sum(p) = 1}, as a list of lists.

    Every vertex has all coordinates at a bound of the interval except at most one, which is at 1 - leftover. 
    
    The DFS decides per coordinate whether it sits at its upper bound and prunes branches without a vertex.
    For k successors, the DFS call is O(k) per vertex and never emits a duplicate.
    """
    l, u = list(lower), list(upper)
    r = 1.0 - sum(l)
    if r < -tol or sum(u) < 1.0 - tol or any(a > b + tol for a, b in zip(l, u)):
        raise ValueError(f"empty polytope for intervals {list(zip(l, u))}")
    r = max(r, 0.0)
    idx = [i for i in range(len(l)) if u[i] - l[i] > tol]    # zero-width coordinates stay fixed
    w = [u[i] - l[i] for i in idx]
    m = len(w)
    suffix = [0.0] * (m + 1)
    for j in range(m - 1, -1, -1):
        suffix[j] = suffix[j + 1] + w[j]
    in_S, out = [False] * m, []

    def dfs(j, s, max_excluded):
        if s > r + tol: # deadend: s overshoots the slack
            return
        leftover = r - s - suffix[j]
        if leftover > tol and max_excluded <= leftover + tol:
            return # deadend: nothing can absorb the leftover
        if j < m:
            in_S[j] = True
            dfs(j + 1, s + w[j], max_excluded)
            in_S[j] = False
            dfs(j + 1, s, max(max_excluded, w[j]))
            return
        p = l[:]
        for i in range(m):
            if in_S[i]:
                p[idx[i]] = u[idx[i]]
        gap = r - s
        if gap <= tol:
            out.append(p) # every coordinate at a bound
            return
        for f in range(m):
            if not in_S[f] and w[f] > gap + tol: # f takes the leftover, strictly inside
                q = p[:]
                q[idx[f]] += gap
                out.append(q)

    dfs(0, 0.0, 0.0)
    return out


def extract_data_from_model(model):
    """All transition data as numpy arrays, via Storm's UMB export (no per-entry Python loop)."""
    opts = stormpy.io.UmbExportOptions()
    opts.compression = stormpy.io.CompressionMode.NO_COMPRESSION
    opts.canonicize_pomdp = False                           # keep Storm's state/choice indices
    opts.value_type = stormpy.io.UmbExportValueType.DOUBLE_INTERVAL
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.umb")
        stormpy.io.export_to_umb(model, path, options=opts)
        with tarfile.open(path) as tar:
            names = set(tar.getnames())

            def read(name, dtype):
                return np.frombuffer(tar.extractfile(name).read(), dtype=dtype)

            n_states = json.load(tar.extractfile("index.json"))["transition-system"]["#states"]
            prob = read("branch-to-probability.bin", "<f8").reshape(-1, 2)
            offsets = read("actions/choices/string-mapping.bin", "<u8")
            raw = tar.extractfile("actions/choices/strings.bin").read()
            return dict(
                state_to_choice=(read("state-to-choices.bin", "<u8").astype(np.int64) if "state-to-choices.bin" in names
                     else np.arange(n_states + 1, dtype=np.int64)),
                choice_to_branches=read("choice-to-branches.bin", "<u8").astype(np.int64),
                target=read("branch-to-target.bin", "<u8").astype(np.int64),
                lower=prob[:, 0].copy(), upper=prob[:, 1].copy(),
                action=read("actions/choices/values.bin", "<u4").astype(np.int32),
                action_names=[raw[offsets[i]:offsets[i + 1]].decode() for i in range(len(offsets) - 1)],
                observations=(read("observations/states/values.bin", "<u8").astype(np.int64)
                              if "observations/states/values.bin" in names else None))

def row_keys(table, base):
    """One comparable key per row of a non-negative integer table, so that equal rows get
    equal keys and np.unique can work on a 1-D array (much faster than np.unique(axis=0)).
 
    A row (a, b, c) with entries below `base` is read as the number a*base^2 + b*base + c.
    If that number could overflow int64, the row's raw bytes are used as the key instead.
    """
    k = table.shape[1]
    if k * np.log2(max(base, 2)) < 62:
        place_values = base ** np.arange(k - 1, -1, -1, dtype=np.int64)
        return table @ place_values
    table = np.ascontiguousarray(table)
    return table.view(np.dtype((np.void, table.itemsize * k))).ravel()


def group_intervals(choice_to_branches, lower, upper):
    """Group choices with the same intervals into patterns, so each polytope is solved once.
 
    Two choices whose branches carry the same intervals (in any order) have the same polytope; only the successor states differ. 
    For example, the choices
        [0.6,0.9] -> t3, [0.1,0.4] -> t7     
    and     
        [0.1,0.4] -> t2, [0.6,0.9] -> t5
    share the pattern ([0.1,0.4], [0.6,0.9]).
 
    Steps:
      1. number the distinct intervals -> every branch gets an interval id;
      2. sort the branches of each choice by interval id (a canonical order);
      3. choices whose sorted id rows are equal share a pattern.
    Choices with the same number of branches k are handled together as a (choices x k) table.
    Every step is one numpy operation instead of a Python loop over choices.
 
    Returns
      order:          branch permutation putting each choice's branches in canonical order;
                      column i of a pattern's vertices belongs to successor target[order][i]
      choice_pattern: pattern index of every choice
      patterns:       (lower bounds, upper bounds) of every pattern, in canonical order
    """
    n_branches = np.diff(choice_to_branches)
 
    # Step 1: interval ids. Writing [l, u] as the complex number l + iu lets one 1-D
    # np.unique find the distinct intervals, numbered in (l, u) order.
    intervals, interval_id = np.unique(lower + 1j * upper, return_inverse=True)
    interval_id = interval_id.ravel()
 
    order = np.arange(len(lower), dtype=np.int64)
    choice_pattern = np.empty(len(n_branches), dtype=np.int64)
    patterns = []
    for k in map(int, np.unique(n_branches)):
        choices = np.flatnonzero(n_branches == k)
        positions = choice_to_branches[choices, None] + np.arange(k) # (choices x k) branch indices
 
        # Step 2: canonical order, i.e. each row sorted by interval id.
        by_interval = np.argsort(interval_id[positions], axis=1, kind="stable")
        sorted_branches = np.take_along_axis(positions, by_interval, axis=1)
        order[positions] = sorted_branches
        ids = interval_id[sorted_branches] # (choices x k) sorted interval ids
 
        # Step 3: equal rows of ids = same pattern.
        _, first, row_pattern = np.unique(row_keys(ids, len(intervals)), return_index=True, return_inverse=True)
        choice_pattern[choices] = len(patterns) + row_pattern.ravel()
        for row in ids[first]: # one representative per new pattern
            patterns.append((intervals[row].real.tolist(), intervals[row].imag.tolist()))
    return order, choice_pattern, patterns



@dataclass
class VertexTable:
    """Vertices of every (s, a) polytope, stored once per distinct interval pattern.

    Choice c (Storm's row index) of state s lies in state_choice_offsets[s]:state_choice_offsets[s+1];
    its successors are branch_target[choice_branch_offsets[c]:choice_branch_offsets[c+1]] and its
    vertices are the rows of pattern_vertices[choice_pattern[c]], with columns in that successor order.

    Sharing: every distinct probability vector is stored once in unique_vertices; the vertices of
    pattern p are unique_vertices[i] for i in pattern_vertex_ids[p]. nr_vertex_choices[c] is the
    number of vertices (choices for nature) of choice c.
    """
    state_choice_offsets: np.ndarray
    choice_action: np.ndarray
    action_names: list
    choice_branch_offsets: np.ndarray
    branch_target: np.ndarray
    branch_lower: np.ndarray
    branch_upper: np.ndarray
    choice_pattern: np.ndarray
    pattern_vertices: list                                  # arrays of shape (n_vertices, k)
    unique_vertices: list                                   # distinct probability vectors
    pattern_vertex_ids: list                                # per pattern: ids into unique_vertices
    nr_vertex_choices: np.ndarray                           # per choice: number of vertices
    observations: np.ndarray | None
    initial_states: np.ndarray

    @property
    def nr_states(self):
        return len(self.state_choice_offsets) - 1

    @property
    def nr_unique_vertices(self):
        return len(self.unique_vertices)

    def vertex_ids(self, s, action):
        """Global ids (into unique_vertices) of the vertices of (s, action)."""
        return self.pattern_vertex_ids[self.choice_pattern[self._choice(s, action)]]

    def nr_choices_of(self, s, action):
        """Number of vertices of (s, action), i.e. how many choices nature has there."""
        return int(self.nr_vertex_choices[self._choice(s, action)])

    def choices_of_state(self, s):
        return range(int(self.state_choice_offsets[s]), int(self.state_choice_offsets[s + 1]))

    def choice_vertices(self, c):
        """(successor states, vertex matrix) of global choice index c."""
        b0, b1 = self.choice_branch_offsets[c], self.choice_branch_offsets[c + 1]
        return self.branch_target[b0:b1], self.pattern_vertices[self.choice_pattern[c]]

    def _choice(self, s, action):
        """Global choice index for an action name or a local choice index of state s."""
        if isinstance(action, (int, np.integer)):
            return self.choices_of_state(s)[action]
        matches = [c for c in self.choices_of_state(s) if self.action_names[self.choice_action[c]] == action]
        if len(matches) != 1:
            raise KeyError(f"state {s} has {len(matches)} choices labelled {action!r}; use a local index")
        return matches[0]

    def vertices(self, s, action):
        """Vertices of state s for an action name or a local choice index."""
        return self.choice_vertices(self._choice(s, action))

    def items(self):
        """Yield (state, local choice index, action name, successors, vertices)."""
        for s in range(self.nr_states):
            for choice, c in enumerate(self.choices_of_state(s)):
                yield (s, choice, self.action_names[self.choice_action[c]], *self.choice_vertices(c))


def compute_vertex_table(model, tol=1e-12):
    """Vertex table of an already built stormpy interval model."""
    d = extract_data_from_model(model)
    order, choice_pattern, patterns = group_intervals(d["choice_to_branches"], d["lower"], d["upper"])
    vertices = [np.array(find_extreme_points(lo, hi, tol), dtype=float).reshape(-1, len(lo)) for lo, hi in patterns]

    # Give each distinct probability vector one global id (patterns are few, so this is cheap).
    # Vectors equal up to tol share an id; the key rounds to a grid of that size.
    digits = max(0, int(-np.log10(tol))) if tol > 0 else 15
    ids, unique, pattern_vertex_ids = {}, [], []
    for V in vertices:
        row_ids = []
        for row in V:
            key = tuple(round(float(x), digits) for x in row)
            if key not in ids:
                ids[key] = len(unique)
                unique.append(row)
            row_ids.append(ids[key])
        pattern_vertex_ids.append(np.array(row_ids, dtype=np.int64))
    nr_vertex_choices = np.array([len(V) for V in vertices], dtype=np.int64)[choice_pattern]

    return VertexTable(
        state_choice_offsets=d["state_to_choice"], choice_action=d["action"], action_names=d["action_names"],
        choice_branch_offsets=d["choice_to_branches"], branch_target=d["target"][order],
        branch_lower=d["lower"][order], branch_upper=d["upper"][order], choice_pattern=choice_pattern,
        pattern_vertices=vertices, unique_vertices=unique, pattern_vertex_ids=pattern_vertex_ids,
        nr_vertex_choices=nr_vertex_choices, observations=d["observations"],
        initial_states=np.array(model.initial_states, dtype=np.int64))

if __name__ == "__main__":
    # MODEL_PATH = "intercept-large.prism"
    import sys
    MODEL_PATH = sys.argv[1]
    program = stormpy.parse_prism_program(MODEL_PATH)
    options = stormpy.BuilderOptions()
    options.set_build_choice_labels(True)
    model = stormpy.build_model[stormpy.Interval](program, options=options)
    table = compute_vertex_table(model)
    
    for s, choice, action, sucessors, vertices in table.items():
        print(f"{s=}, {choice=}, {action=}, {sucessors=}, {vertices=}")

    print("Done.")
    
    # Gather single vertex (of the last (s, a) pair listed above)
    sucessors, V = table.vertices(s, action)
    print(sucessors, V)
    # Finds vertices from global choice index.
    print(table.choice_vertices(table._choice(s, action)))
    v = V[0]
    print(f"first vertex of ({s}, {action}): {v}, global id {table.vertex_ids(s, action)[0]}, "
          f"{table.nr_choices_of(s, action)} choice(s)")

    print(f"Unique vertices: {table.nr_unique_vertices}")
    print(f"Max number of vertices for nature to choose from across (s,a): {table.nr_vertex_choices.max()}")
    print(f"Total num vertices (sum of choices over all (s,a)): {table.nr_vertex_choices.sum()} ({table.nr_vertex_choices.max()})")
    counts, freq = np.unique(table.nr_vertex_choices, return_counts=True)
    print("Choices per (s,a): " + ", ".join(f"{c} -> {f} pairs" for c, f in zip(counts, freq)))
    
    