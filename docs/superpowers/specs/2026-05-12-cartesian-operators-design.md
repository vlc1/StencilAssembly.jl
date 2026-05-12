# CartesianOperators.jl — Design Spec

Date: 2026-05-12

## Purpose

Build a Julia package that, given a stencil pattern (list of mesh-space
`CartesianIndex` offsets), a row-side `CartesianRunIndices{N}`, and a
column-side `CartesianRunIndices{N}`, assembles a `SparseMatrixCSC` representing
the masked stencil operator. The two coordinate spaces are mirrored from
[CartesianRuns](https://github.com/vlc1/CartesianRuns.jl/) (mask space →
mesh-side `CartesianIndex`; compact space → 1-based linear position of the
active cells), and `CartesianRunIndices` provides the mapping between them.

The package targets sparse Jacobian assembly for CFD-style stencil operators on
masked Cartesian meshes, where the row mask (ψ) and column mask (φ) may differ.
"Implicit Jacobian" and "explicit Jacobian" reduce to two calls with different
row/column masks; the package itself has no notion of ψ vs φ.

## Non-goals (v1)

- Variable coefficients (functions of position). v1 assumes coefficients
  constant across the mesh.
- Automatic differentiation of nonlinear `F`. The user supplies stencil + coefs.
- GPU support.
- Automatic stencil discovery from a residual expression.

## API surface — two-stage

The package exposes the sparsity-pattern construction and the numerical fill as
two separate operations. The numeric fill is allocation-free so it can be
re-run many times inside an outer iterative solver.

```julia
# Symbolic pass — allocates colptr/rowval; run once per (mask, stencil) combo.
J = sparsity_pattern(stencil, row_cri, col_cri, ::Type{T} = Float64)
    -> SparseMatrixCSC{T,Int}   # colptr, rowval populated; nzval undef

# Numeric pass — allocation-free, in-place on J.nzval. Reusable.
fill_values!(J, stencil, coefs, row_cri, col_cri) -> J
```

The exact shape of the symbolic output (plain `SparseMatrixCSC` vs a wrapper
carrying a "fill plan") is **deferred** — see Open Questions.

## Boundary policy

- A stencil offset whose mesh image lies outside the column mask → that single
  `(row, col)` entry is dropped from the pattern.
- Rows are sourced from `row_cri` itself, so rows outside the row mask are
  never emitted (automatic by iteration).

## Kernel design

The implementation mirrors CartesianRuns's three-layer dispatch.

| Layer       | Pattern                                              | Role                                                |
| ----------- | ---------------------------------------------------- | --------------------------------------------------- |
| Public API  | `sparsity_pattern(stencil, row, col)`                | type setup, top-level recursion, output allocation  |
| N-D kernel  | `_stencil_fused!(...)`                               | recursive dimensional peeling, two-pointer sweep    |
| 1-D kernel  | `_stencil_runs!(...)`                                | flat pointer sweep over `intervals[1]` of both cri  |

### Why not `Base.in` / `getindex`?

`Base.in(idx, cri)` does an `O(log R)` `searchsortedfirst` per dimension, so
per-cell membership testing costs `O(nnz · N · log R)` total and ignores the
interval structure. The package must instead use **pointer-based sweeps** over
the underlying `intervals` and `offsets` arrays — the same style as
`_intersect_runs!` / `_intersect_fused!` — with all compact-index arithmetic
falling out of `Interval.shift`.

### Pointer-based 1-D sweep

The 1-D kernel walks `col_cri.intervals[1]` with a single pointer; for each
stencil offset Δ it maintains an independent monotone pointer into
`row_cri.intervals[1]`. At each step, overlap of `(col_iv.mask,
row_iv.mask .+ Δ)` is computed with `max`/`min`:

```
lo = max(col_iv.mask.start, row_iv.mask.start + Δ)
hi = min(col_iv.mask.stop,  row_iv.mask.stop  + Δ)
```

The overlap `[lo..hi]` yields:

- `col_compact = (lo : hi)             .+ shift(col_iv)`
- `row_compact = ((lo - Δ) : (hi - Δ)) .+ shift(row_iv)`

No `Base.in` or `getindex` queries. Pointer advancement follows the
`_intersect_runs!` convention: advance whoever ends first
(`col_iv.mask.stop ≤ row_iv.mask.stop + Δ` → advance col; else advance row).

### Offset ordering for CSC sortedness

SparseMatrixCSC requires `rowval` ascending within each column. If the stencil
offsets are sorted **descending in column-major lexicographic order** at the
public-API boundary, then per column the row entries emitted by the sweep fall
out monotonically increasing — `rowval` is naturally sorted without any
per-column sort step. The comparator for offsets is column-major lex on
`CartesianIndex` (innermost dimension varies fastest, descending).

### N-D recursion

`_stencil_fused!` mirrors `_intersect_fused!`:

- Outer dimensions: two-pointer sweep over `row_cri.intervals[d]` and
  `col_cri.intervals[d]` with a shift Δ_d applied to the row-side mask range
  (exactly the 1-D pointer pattern, just at the outer level). Overlap detection
  uses `_inner_slice` and `view` to pass cost-free `SubArray`s of inner
  interval vectors to the recursive call. State variables (`prev`, `start`,
  `n_d`, `m_d`, etc.) follow CartesianRuns's naming.
- Bottom dispatch (`NTuple{1,...}` + `Tuple{}` offsets): call `_stencil_runs!`.

Dimensions where Δ_d = 0 collapse to a vanilla two-pointer intersection sweep.

### Symbolic vs numeric: shared sweep, different emission action

Both passes execute the **identical** sweep — only the action at each emission
step differs.

- Symbolic: append `row_compact` slot(s) to `rowval`, advance per-column counter.
- Numeric: write `nzval[colptr[c] + offset_local_idx]` for the corresponding
  column slot.

This makes "re-walk, no fill plan" the natural default. Whether to materialise
an auxiliary fill plan in the symbolic output is **deferred** (see Open
Questions).

## Staged delivery plan

Hard-coded operators first; generic `Stencil`-based API only after the kernel
pattern is visible across at least three operator families. Each phase ends in
a review checkpoint.

### Phase 0 — Package scaffold

PkgTemplates-generated scaffold under `CartesianOperators.jl` (Project.toml,
src/, test/, .github/workflows/CI.yml, LICENSE). Add `CartesianRuns` and
`SparseArrays` to deps. Smoke test: package loads. Author an initial
`AGENTS.md` mirroring CartesianRuns's structure.

### Phase 1 — Differences along x (dim 1)

Each operator ships as a **pair**:
`_pattern(row_cri, col_cri) -> SparseMatrixCSC`,
`_fill!(J, row_cri, col_cri) -> J`. Offsets and coefficients are hard-coded
per operator.

**Phase 1a — 1-D.** Six functions:
`forward_x_pattern_1d`, `forward_x_fill_1d!`, `backward_x_*`, `central_x_*`.
Tests compare against a brute-force dense reference built by `expand`-ing both
cri's and applying the stencil cellwise.

→ **Review checkpoint #1.**

**Phase 1b — N-D.** Same operators, generic over `N`. Outer dims peel with
zero shift (intersection-only sweep); bottom dispatches to the 1-D kernel.

→ **Review checkpoint #2.**

### Phase 2 — Differences along y (dim 2)

Same pair pattern; first operators with a **non-zero outer-dim shift** —
exercises the full outer sweep with Δ_d ≠ 0.

**Phase 2a — 2-D.** **Phase 2b — N-D.**

→ **Review checkpoint #3.**

### Phase 3 — Laplacian

**Phase 3a — 1-D** (offsets `{-1, 0, +1}`, three coefs).
**Phase 3b — N-D** (`2N+1` offsets). First operator combining inner-dim and
outer-dim shifts; first to need the descending-lex offset sort to keep
`rowval` per-column ordering correct.

→ **Review checkpoint #4 + revisit Open Question Q3.**

### Phase 4 — Generalize

With three operator families implemented, extract:

- A `Stencil` type (sorted offsets + permutation back to caller order).
- The final shape of `sparsity_pattern` / `fill_values!`, including the
  symbolic-output structure (plain `SparseMatrixCSC` vs wrapper-with-plan).
- Consolidated `_stencil_runs!` / `_stencil_fused!` replacing the hard-coded
  per-operator code.

## Open questions

- **Q3 (deferred to Phase 4)**: should `sparsity_pattern` return a plain
  `SparseMatrixCSC` and `fill_values!` re-walk the cri, or a wrapper carrying a
  fill-plan that makes `fill_values!` a flat `nnz`-long loop? Decision
  postponed until concrete kernel performance is observable.
- **Coefficient ↔ offset ordering** (Phase 4): caller-supplied offset order vs.
  internally sorted (with permutation) — to be resolved when `Stencil` is
  introduced.
- **Variable coefficients**: deferred past v1.

## Anticipated file layout (post-Phase-4)

| File                                  | Role                                                    |
| ------------------------------------- | ------------------------------------------------------- |
| `src/CartesianOperators.jl`           | Module entry; exports + `include`s                      |
| `src/types.jl`                        | `Stencil` (Phase 4); maybe `StencilOperator`            |
| `src/sparsity.jl`                     | `sparsity_pattern`, `_stencil_fused!`, `_stencil_runs!` |
| `src/fill.jl`                         | `fill_values!`, `_fill_fused!`, `_fill_runs!`           |
| `src/x_diff.jl`                       | Phase 1 hard-coded operators (1-D + N-D)                |
| `src/y_diff.jl`                       | Phase 2 hard-coded operators (2-D + N-D)                |
| `src/laplacian.jl`                    | Phase 3 hard-coded operators (1-D + N-D)                |
| `test/runtests.jl`                    | Test suite                                              |
| `AGENTS.md`                           | Mirrors CartesianRuns conventions                       |

Generic `Stencil`-based code (`types.jl`, `sparsity.jl`, `fill.jl`) lands in
Phase 4; the per-operator files remain for regression tests against the
generic implementation.
