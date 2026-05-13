# AGENTS.md

## Project goal

Build a Julia package that assembles `SparseMatrixCSC` operators for stencil
patterns on masked Cartesian meshes, where row and column masks are
`CartesianRuns.CartesianRunIndices{N}`. Sparsity pattern and numerical fill are
exposed as separate operations; the fill is allocation-free up to a small
constant-size buffer.

- Package name: **`CartesianOperators.jl`**.
- Reference design: `docs/superpowers/specs/2026-05-12-cartesian-operators-design.md`.

## Files

| File                          | Role                                              |
| ----------------------------- | ------------------------------------------------- |
| `src/CartesianOperators.jl`   | Module entry; imports; `include`s; exports        |
| `src/x_diff.jl`               | x-direction difference kernels and wrappers       |
| `test/runtests.jl`            | Test suite entry (run via `Pkg.test()`)           |
| `test/reference.jl`           | Brute-force `stencil_reference` helper            |
| `test/oracle.jl`              | `spdiagm`-based naive cross-check oracle          |
| `test/test_x_diff.jl`         | x-diff test sets                                  |

## Sticky decisions (do not re-litigate)

### Two-stage API

- `*_pattern(row_cri, col_cri, T = Float64) -> SparseMatrixCSC{T,Int}` builds
  `colptr`/`rowval`; `nzval` is allocated but undef.
- `*_fill!(J, row_cri, col_cri) -> J` writes `nzval`, allocation-free apart
  from a small per-call buffer (size = number of stencil offsets).

### Boundary policy

Stencil offset → column outside `col_cri` ⇒ that single `(row, col)` entry is
dropped. Rows are sourced from `row_cri` so off-mask rows are never emitted.

### CartesianRunIndices surface we depend on

`CartesianRunIndices` is treated as a minimal value carrying only:

- `cri.intervals::NTuple{N,<:AbstractVector{Interval}}`
- `cri.offsets::NTuple{N-1,<:AbstractVector{Int}}` (empty tuple for `N = 1`)
- `length(cri)` (compact-space size)
- `iterate(cri)` / `cri[k]` (used only in test oracles, never in kernels)

We deliberately do **not** depend on `domain(cri)` or a `domain` field — those
no longer exist on the current CartesianRuns `ghost` branch. The kernels work
on raw mesh integers extracted from `Interval.mask` ranges and `shift(::Interval)`;
the matrix structure is determined by these alone. `row_cri` and `col_cri` are
interpreted on a shared integer mesh, and the caller owns the coherence of
that interpretation.

### Pointer-based sweep, no `Base.in` / `getindex`

Kernels walk `cri.intervals[d]` directly with pointers (one per stencil offset
in dim 1). Compact indices come from `Interval.shift` via range arithmetic.
Mirrors `_intersect_runs!` / `_intersect_fused!` from CartesianRuns.

### Offset ordering for CSC sortedness

Stencil offsets are sorted **descending in column-major lex** so per column the
emitted rowval is monotonically ascending without a separate sort step.

## Conventions

- 1-D base kernel name pattern: `_<op>_pattern_runs!`, `_<op>_fill_runs!`.
- N-D recursive kernel name pattern: `_<op>_pattern_fused!`, `_<op>_fill_fused!`.
- Hard-coded wrapper name pattern: `forward_x_pattern`, `forward_x_fill!`, etc.
- Run tests: `julia --project=. -e 'using Pkg; Pkg.test()'` from package root.
