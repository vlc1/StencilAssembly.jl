# CartesianOperators

[![Build Status](https://github.com/vlc1/CartesianOperators.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/vlc1/CartesianOperators.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia package that assembles `SparseMatrixCSC` operators for stencil patterns
on masked Cartesian meshes, where row and column masks are
[`CartesianRuns.CartesianRunIndices{N}`](https://github.com/vlc1/CartesianRuns.jl/).
Sparsity pattern construction and numerical fill are exposed as separate
operations; the fill is allocation-free up to a small constant-size buffer, so
it can be re-run cheaply inside an outer iterative solver.

```julia
using CartesianOperators, CartesianRuns

row = CartesianRunIndices(Bool[1, 0, 1, 1])
col = CartesianRunIndices(Bool[1, 1, 0, 1])

# A stencil = (mesh dimension, offsets, coefs). Forward x-difference:
forward = LinearStencil{1}((1, 0), (1.0, -1.0))   # (D ϕ)[i] = ϕ[i+1] − ϕ[i]

# assemble builds the sparsity pattern; update! fills the values. They are
# split so the pattern is built once and the values re-filled cheaply across
# many iterations of an outer solver.
J = assemble(forward, row, col)   # SparseMatrixCSC{Float64,Int}, nzval undef
update!(J, forward, row, col)     # writes J.nzval allocation-free
```

## How operations work

### `LinearStencil{D,K,T}`

A `LinearStencil{D,K,T}` carries everything an operator needs:

- `D::Int` — mesh dimension on which the stencil acts (1-based).
- `K::Int` — number of stencil terms.
- `T` — coefficient eltype.
- `offsets::NTuple{K,Int}` — strictly descending 1-D offsets along dim `D`.
- `coefs::NTuple{K,T}` — matching coefficients.

The three classical first-order x-differences are:

| Operator   | `LinearStencil{1}` constructor          | Stencil                            |
|------------|-----------------------------------------|------------------------------------|
| Forward x  | `LinearStencil{1}((1,  0), (1.0, -1.0))`| `(D ϕ)[i] = ϕ[i+1] − ϕ[i]`         |
| Backward x | `LinearStencil{1}((0, -1), (1.0, -1.0))`| `(D ϕ)[i] = ϕ[i]   − ϕ[i-1]`       |
| Central x  | `LinearStencil{1}((1, -1), (1.0, -1.0))`| `(D ϕ)[i] = ϕ[i+1] − ϕ[i-1]`       |

### Two-stage operations

| Function    | Allocates? | Does what                                                            |
|-------------|------------|----------------------------------------------------------------------|
| `assemble(st, row, col)` | yes | builds `colptr` and `rowval`; allocates `nzval` but leaves it undef |
| `update!(J, st, row, col)` | no (apart from a `K`-element buffer) | writes `J.nzval` in place |

Symbolic structure is computed once per `(stencil, masks)` triple; numeric fill
is reused many times in an outer non-linear solve.

### Subtraction convention

For each offset `Δ`, the matrix entry from column `c` (at mesh position `p_c`)
lands on the row at mesh position `p_c − Δ`. Equivalently, row `r` (at mesh
`p_r`) contributes at column `p_r + Δ`. This is the **subtraction
convention**: the offset goes *from row to column*. It matches the
mathematical reading of `(D ϕ)[i] = … ϕ[i + Δ] …`, where the row index is `i`
and the column index is `i + Δ`.

### Pointer-based sweep over interval vectors

Kernels walk `cri.intervals[d]` directly with pointers — one for the column
side and one per stencil offset on the row side. Compact indices fall out of
`Interval.shift` via range arithmetic; the kernel never calls `Base.in` or
`Base.getindex` on the `CartesianRunIndices`. This mirrors the style of
`_intersect_runs!` / `_intersect_fused!` in
[CartesianRuns](https://github.com/vlc1/CartesianRuns.jl/), and is the
performance core: cost scales with the number of runs (and the stencil width),
not with the number of `true` cells.

### No domain match required

`row` and `col` are interpreted on a shared integer mesh. Nothing in the
kernel or operations reads `domain(row)` / `domain(col)` — the matrix
structure depends only on each cri's `intervals` field and the stencil's
`offsets`. The caller owns the coherence of that interpretation (typically:
both cri's come from masks of the same physical grid). If the cri's were
constructed from masks on disjoint integer ranges, the resulting matrix is
simply empty (no stencil offset bridges the two), which is mathematically
correct.

### Offset ordering is tied to `SparseMatrixCSC`

`SparseMatrixCSC` requires `rowval` to be sorted ascending within each column.
We satisfy that **without an explicit per-column sort** by requiring stencil
offsets to be **strictly descending** at the type boundary: under the
subtraction convention, for column `c` and offset `Δ`, the row mesh-position
is `c − Δ`, so iterating offsets descending makes the emitted rows
monotonically ascending in compact space.

This invariant is enforced by the `LinearStencil` inner constructor via
`issorted(offsets; lt = >=)`, throwing `ArgumentError` on ascending or
duplicate offsets. The check is intrinsic to CSC; for CSR the constraint
would flip to ascending; for COO with a deferred sort it disappears.

## Status

This is a work in progress. Currently implemented:

- `LinearStencil{D,K,T}` type (any `D`; strict-descending offsets).
- `assemble` / `update!` for `LinearStencil{1}` against
  `CartesianRunIndices{1}` (1-D only).

Next milestone: N-D dispatch — `LinearStencil{D}` against
`CartesianRunIndices{N}` for any `D ≤ N` via a recursive dimensional-peeling
kernel; then a higher-level abstraction for compositions (e.g., the
Laplacian as a sum of `LinearStencil`s along each dimension).

### Further reading

- [`docs/plan.md`](docs/plan.md) — forward-looking implementation plan
  (status, roadmap, next-milestone design).
- [`docs/superpowers/specs/2026-05-12-cartesian-operators-design.md`](docs/superpowers/specs/2026-05-12-cartesian-operators-design.md)
  — original design rationale.
