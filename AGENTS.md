# AGENTS.md

## Project goal

Build a Julia package that assembles `SparseMatrixCSC` operators for stencil
patterns on masked Cartesian meshes, where row and column masks are
`CartesianRuns.CartesianRunIndices{N}`. Sparsity pattern and numerical fill are
exposed as separate operations; the fill is allocation-free up to a small
constant-size buffer.

- Package name: **`CartesianOperators.jl`**.
- Forward-looking plan: [`docs/plan.md`](docs/plan.md).
- Design rationale (historical): [`docs/superpowers/specs/2026-05-12-cartesian-operators-design.md`](docs/superpowers/specs/2026-05-12-cartesian-operators-design.md).

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

The package's main abstraction is
`LinearStencil{D,K,T,N,C<:NTuple{K,AbstractArray{T,N}}}`:

- `D::Int` — mesh dimension the stencil acts on (1-based).
- `K::Int` — number of stencil terms.
- `T` — shared element type of every coef array.
- `N` — coef-array dimensionality; matches `CartesianRunIndices{N}` at
  assembly time.
- `offsets::NTuple{K,Int}` — strictly descending 1-D offsets along dim `D`.
- `coefs::C` — `NTuple{K,<:AbstractArray{T,N}}` of coefficient arrays.
  Heterogeneous containers (e.g. `Fill` + `Vector` + `OffsetArray`) are
  fine as long as they share `eltype` and `ndims`.

The inner constructor `LinearStencil{D}(offsets, coefs)` (well-typed path)
validates `D ≥ 1` and strict-descending offsets (`issorted(offsets; lt = >=)`).
A catch-all outer constructor `LinearStencil{D}(::Tuple, ::Tuple)` reports
friendly `ArgumentError`s for ill-typed inputs (length mismatch, non-`Int`
offsets, non-`AbstractArray` coefs, mixed `eltype`, mixed `ndims`). The
shared `eltype` and `ndims` of well-typed coefs are enforced at the method
signature, not in the constructor body.

Two operations:

- `assemble(st, row, col, ::Type{SparseMatrixCSC{T,Int}} = SparseMatrixCSC{T,Int})`
  — builds `colptr`/`rowval`; allocates `nzval` (uninitialised).
- `update!(mat, st, row, col) -> mat` — writes `nzval` in place; allocation-
  free apart from a `K`-element scratch buffer and whatever `getindex` on
  the user's coef arrays costs (`Vector`, `Fill`, `OffsetArray` are O(1)).

The trailing `::Type` argument on `assemble` is positional for future matrix-
format extension; v1 supports only `SparseMatrixCSC{T,Int}`.

### Coefficient indexing: column anchor on the shared mesh

`row` and `col` `CartesianRunIndices` are interpreted on a *single shared*
integer mesh — there is no separate "row mesh" or "column mesh" coordinate
system, only different active subsets. Each `coefs[k]` is an
`AbstractArray{T,N}` that lives on that shared mesh and is indexed by a
`CartesianIndex{N}` mesh position.

At each emission, the kernel has both the column's mesh position `c_idx`
and the row's mesh position `r_idx = c_idx − Δ_idx`. By convention it
reads `coefs[k][c_idx]` — the **column anchor**. The choice is symmetric
in cost (the kernel computes `r_idx` anyway for row-pointer arithmetic)
and matches the CSC sweep order where the column index is the outer loop.

For non-square operators (e.g., staggered grids, restriction operators)
the coef "naturally" attached to row position `r_idx` for some slots must
be supplied as an array that, when indexed at `c_idx`, yields the value
the user intends for the entry `(r=c−Δ, c)`. The typical recipe is
`OffsetArrays.OffsetArray` to compensate, or to construct the coef array
directly in mesh-space such that `coefs[k][c_idx]` evaluates to the
intended weight. The kernel does not validate axes — accessing outside
the supplied coef array's axes raises `BoundsError` at the emission site.

For constant coefficients use `FillArrays.Fill(value, axes)` — O(1) storage
and O(1) `getindex`.

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
no longer exist on the current CartesianRuns `main` (the `ghost` branch was
merged and deleted). The kernels work on raw mesh integers extracted from
`Interval.mask` ranges and `shift(::Interval)`; the matrix structure is
determined by these alone. `row` and `col` are interpreted on a shared
integer mesh, and the caller owns the coherence of that interpretation.

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

Implemented: `LinearStencil{D,K,T,N,C}` constructor (any `D`, any `N`) with
variable coefficients as `AbstractArray{T,N}`; `assemble` and `update!` for
1-D dispatch only (`LinearStencil{D,K,T,1}` against `CartesianRunIndices{1}`
→ `SparseMatrixCSC{T,Int}`).

Next milestone (see [`docs/plan.md`](docs/plan.md) for the design):

- **N-D dispatch** — `CartesianRunIndices{N}` × `LinearStencil{D,K,T,N}` with
  `1 ≤ D ≤ N` via recursive dimensional-peeling kernels
  (`_pattern_fused!`, `_fill_fused!`). The new kernels branch on `Nd vs
  D` and bottom out at `_pattern_runs!` / `_pattern_runs_intersect!`
  depending on whether the base case is the stencil dim or an inner
  intersection dim.

Further deferred milestones (sketched in the plan): composition (e.g.,
Laplacian as a sum of `LinearStencil`s), non-`SparseMatrixCSC` matrix
targets (`BandedMatrix`, dense), and a contiguous-offset kernel fast path
(single row-pointer march per cell when offsets form a contiguous descending
range).
