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

| File                          | Role                                                          |
| ----------------------------- | ------------------------------------------------------------- |
| `src/CartesianOperators.jl`   | Module entry; imports; `include`s; exports                    |
| `src/stencil.jl`              | `LinearStencil`, `assemble`, `update!`, 1-D pointer kernels   |
| `test/runtests.jl`            | Test suite entry (run via `Pkg.test()`)                       |
| `test/reference.jl`           | Brute-force `stencil_reference` helper                        |
| `test/oracle.jl`              | `spdiagm`-based naive cross-check oracle                      |
| `test/test_stencil.jl`        | LinearStencil + assemble + update! test sets                  |

## Sticky decisions (do not re-litigate)

### Type-driven API

The package's main abstraction is `LinearStencil{D,K,T}`:

- `D::Int` — mesh dimension the stencil acts on (1-based).
- `K::Int` — number of stencil terms.
- `T` — coefficient eltype.
- `offsets::NTuple{K,Int}` — strictly descending 1-D offsets along dim `D`.
- `coefs::NTuple{K,T}` — matching coefficients.

The inner constructor `LinearStencil{D}(offsets, coefs)` validates `D ≥ 1`
and strict-descending offsets (`issorted(offsets; lt = >=)`).

Two operations:

- `assemble(st, row, col, ::Type{SparseMatrixCSC{T,Int}} = SparseMatrixCSC{T,Int})`
  — builds `colptr`/`rowval`; allocates `nzval` (uninitialised).
- `update!(mat, st, row, col) -> mat` — writes `nzval` in place; allocation-
  free apart from a `K`-element scratch buffer.

The trailing `::Type` argument on `assemble` is positional for future matrix-
format extension; v1 supports only `SparseMatrixCSC{T,Int}`.

### Subtraction convention

For column `c` at mesh position `p_c` and stencil offset `Δ`, the matrix
entry lands on the row at mesh position `p_c − Δ`. Stencil offsets are 1-D
`Int` (along dim `D`).

### Boundary policy

Stencil offset → column outside `col` ⇒ that single `(row, col)` entry is
dropped. Rows are sourced from `row` so off-mask rows are never emitted.

### CartesianRunIndices surface we depend on

`CartesianRunIndices` is treated as a minimal value carrying only:

- `cri.intervals::NTuple{N,<:AbstractVector{Interval}}`
- `cri.offsets::NTuple{N-1,<:AbstractVector{Int}}` (empty tuple for `N = 1`)
- `length(cri)` (compact-space size)
- `iterate(cri)` / `cri[k]` (used only in test oracles, never in kernels)

We deliberately do **not** depend on `domain(cri)` or a `domain` field — those
no longer exist on the current CartesianRuns `ghost` branch. The kernels work
on raw mesh integers extracted from `Interval.mask` ranges and
`shift(::Interval)`; the matrix structure is determined by these alone. `row`
and `col` are interpreted on a shared integer mesh, and the caller owns the
coherence of that interpretation.

### Pointer-based sweep, no `Base.in` / `getindex`

Kernels walk `cri.intervals[d]` directly with pointers (one for col, one per
stencil offset on row). Compact indices come from `Interval.shift` via range
arithmetic. Mirrors `_intersect_runs!` / `_intersect_fused!` from
CartesianRuns.

### Offset ordering for CSC sortedness

`SparseMatrixCSC`'s `rowval`-per-column sortedness is achieved without an
explicit sort by requiring stencil offsets to be **strictly descending** at
the `LinearStencil` constructor boundary. Under the subtraction convention,
descending offsets ⇒ ascending rows per column. The check uses
`issorted(offsets; lt = >=)`. The invariant is intrinsic to CSC; CSR would
flip to ascending.

## Conventions

- 1-D base kernel name pattern: `_pattern_runs!`, `_fill_runs!` (dim-agnostic
  — they take whatever interval vector and offsets they're handed).
- N-D recursive kernel name pattern (deferred): `_pattern_fused!`,
  `_fill_fused!`.
- Public API: `LinearStencil`, `assemble`, `update!`.
- Run tests: `julia --project=. -e 'using Pkg; Pkg.test()'` from package root.

## Scope

Implemented: `LinearStencil{D,K,T}` constructor (any `D`); `assemble` and
`update!` for 1-D dispatch only (`LinearStencil{1}` against
`CartesianRunIndices{1}` → `SparseMatrixCSC{T,Int}`).

Deferred: N-D dispatch (`CartesianRunIndices{N}` × `LinearStencil{D}` with
`1 ≤ D ≤ N`) via a recursive dimensional-peeling kernel; higher-level
abstractions (Laplacian as composition); non-`SparseMatrixCSC` matrix
targets.
