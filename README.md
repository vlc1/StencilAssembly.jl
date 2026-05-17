# CartesianOperators

[![Build Status](https://github.com/vlc1/CartesianOperators.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/vlc1/CartesianOperators.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia package that assembles `SparseMatrixCSC` operators for stencil patterns
on rectangular Cartesian meshes, where row and column index sets are
`NTuple{N, AbstractUnitRange{Int}}`. Sparsity pattern construction and numerical
fill are exposed as separate operations; both kernels are allocation-free, so
the fill can be re-run cheaply inside an outer iterative solver.

```julia
using CartesianOperators, FillArrays

row = (1:4,)
col = (1:4,)

# A stencil = (mesh dimension, offsets, per-term coefficient arrays).
# Forward x-difference: (D ϕ)[i] = ϕ[i+1] − ϕ[i], constant coefs via Fill.
forward = LinearStencil{1}((1, 0), (Fill(1.0, 4), Fill(-1.0, 4)))

# One-shot:
J = build(forward, row, col)         # fully populated SparseMatrixCSC{Float64,Int}

# Or split the pattern/values phases — pattern is built once and the values
# re-filled cheaply across many iterations of an outer solver:
J = assemble(forward, row, col)      # colptr/rowval built; nzval undef
update!(J, forward, row, col)        # writes J.nzval, allocation-free
```

## How operations work

### `LinearStencil{D,K,T,N,C}`

A `LinearStencil` carries everything an operator needs:

- `D::Int` — mesh dimension on which the stencil acts (1-based), `1 ≤ D ≤ N`.
- `K::Int` — number of stencil terms.
- `T` — shared element type of every coef array.
- `N` — coef-array dimensionality; matches the row/col tuple length at
  assembly time.
- `C` — concrete tuple type of the coef containers.
- `offsets::NTuple{K,Int}` — strictly descending 1-D offsets along dim `D`.
- `coefs::C` — `NTuple{K,<:AbstractArray{T,N}}` of coefficient arrays.
  Heterogeneous containers (e.g. `Fill` + `Vector` + `OffsetArray`) are
  fine as long as they share `eltype` and `ndims`.

The three classical first-order x-differences are:

| Operator   | `LinearStencil{1}` constructor                          | Stencil                            |
|------------|---------------------------------------------------------|------------------------------------|
| Forward x  | `LinearStencil{1}((1,  0), (Fill(1.0, n), Fill(-1.0, n)))` | `(D ϕ)[i] = ϕ[i+1] − ϕ[i]`         |
| Backward x | `LinearStencil{1}((0, -1), (Fill(1.0, n), Fill(-1.0, n)))` | `(D ϕ)[i] = ϕ[i]   − ϕ[i-1]`       |
| Central x  | `LinearStencil{1}((1, -1), (Fill(1.0, n), Fill(-1.0, n)))` | `(D ϕ)[i] = ϕ[i+1] − ϕ[i-1]`       |

For variable coefficients, use any `AbstractArray` whose axes cover `col`:

```julia
ρ = rand(n)                                       # density on the mesh
grad = LinearStencil{1}((0, -1), (1 ./ ρ, -1 ./ ρ))  # ψ[i] = (φ[i] − φ[i−1]) / ρ[i]
```

The constructor enforces `D ≥ 1`, `D ≤ N`, strict-descending offsets, and
matching `eltype`/`ndims` across all coef arrays. Ill-typed inputs raise
`ArgumentError` with diagnostic messages.

### Three operations

| Function                   | Allocates? | Does what                                                              |
|----------------------------|------------|------------------------------------------------------------------------|
| `assemble(st, row, col)`   | `colptr` + `rowval` + uninitialised `nzval` | builds the sparsity pattern  |
| `update!(J, st, row, col)` | no         | writes `J.nzval` in place, allocation-free                             |
| `build(st, row, col)`      | same as assemble | convenience: `update!(assemble(st, row, col), st, row, col)` in one shot |

Symbolic structure is computed once per `(stencil, row, col)` triple; numeric
fill is reused many times in an outer non-linear solve. When you don't need
the split, `build` returns a fully populated matrix.

For 1-D, `assemble` and `update!` pin both `D = 1` and `N = 1` at the type
level. Misuse (e.g. `LinearStencil{2}` against 1-D row/col) raises
`MethodError`; the `D ≤ N` invariant at the constructor catches the most
common misuse earlier with a friendly `ArgumentError`.

### Row/col representation: rectangular ranges on a shared mesh

`row` and `col` are `NTuple{N, AbstractUnitRange{Int}}` interpreted on a
*single shared* integer mesh — rectangular sub-blocks of the same mesh. They
may overlap fully, partially, or not at all; they may be unequal in length
or start at different positions (e.g. `row = (1:5,)`, `col = (3:7,)`).

The matrix size is `prod(length, row) × prod(length, col)`. Per dimension,
the compact row index for mesh position `r` is `r − first(row[d]) + 1`, and
analogously for column. Julia's `LinearIndices` / `CartesianIndices` handle
the linear ↔ Cartesian translation internally.

### Subtraction convention

For each offset `Δ`, the matrix entry from column `c` (at mesh position `p_c`)
lands on the row at mesh position `p_c − Δ`. Equivalently, row `r` (at mesh
`p_r`) contributes at column `p_r + Δ`. This matches the mathematical reading
of `(D ϕ)[i] = … ϕ[i + Δ] …`, where the row index is `i` and the column index
is `i + Δ`.

### Column-anchored coefficients

At each emission the kernel reads `coefs[k][c]` — the **column anchor**. Each
`coefs[k]` is an `AbstractArray{T,N}` whose axes must cover `col` (otherwise
indexing raises `BoundsError`). For non-square operators (e.g. staggered grids,
restriction operators) where the coef "naturally" belongs to the row position,
use `OffsetArrays.OffsetArray` to align indexing with mesh positions, or
construct the coef array directly in mesh-space so `coefs[k][c]` returns the
intended weight. For constant coefs use `FillArrays.Fill` — O(1) storage and
O(1) `getindex`.

### Kernel shape: K outer loops + counting-sort-CSC

The kernels run `K` outer loops, one per offset. For each `k`, the c-range
where offset `k` produces an in-range row is computed once:

```
c_lo_k = max(first(col), rmin + offsets[k])
c_hi_k = min(last(col), rmax + offsets[k])
```

The inner loop sweeps `c_lo_k:c_hi_k` and emits unconditionally — no
per-cell bounds check. Per-column "next free slot" is tracked by mutating
`colptr` in place; restored to a proper CSC offset table by a shift-right
pass at the end. Classic counting-sort-CSC, allocation-free.

### Offset ordering is tied to `SparseMatrixCSC`

`SparseMatrixCSC` requires `rowval` to be sorted ascending within each column.
We satisfy that **without a per-column sort** by requiring stencil offsets to
be **strictly descending** at the type boundary: under the subtraction
convention, `r_k = c − offsets[k]` is ascending in `k`, so iterating
`k = 1, …, K` outermost produces rows in ascending order within every
column.

The invariant is enforced by the `LinearStencil` inner constructor via
`issorted(offsets; lt = >=)`, raising `ArgumentError` on ascending or
duplicate offsets. The constraint is intrinsic to CSC; for CSR it would
flip to ascending; for COO with a deferred sort it disappears.

## Status

This is a work in progress. Currently implemented:

- `LinearStencil{D,K,T,N,C}` type (any `1 ≤ D ≤ N`, variable coefficients).
- `assemble`, `update!`, `build` for `LinearStencil{1,K,T,1}` against
  `NTuple{1, AbstractUnitRange{Int}}` (1-D only).

Next milestone: N-D dispatch — `LinearStencil{D}` against
`NTuple{N, AbstractUnitRange{Int}}` for any `1 ≤ D ≤ N` via a recursive
dimensional-peeling kernel; then a higher-level abstraction for compositions
(e.g., the Laplacian as a sum of `LinearStencil`s along each dimension).

### Further reading

- [`docs/plan.md`](docs/plan.md) — forward-looking implementation plan
  (status, roadmap, next-milestone design).
- [`AGENTS.md`](AGENTS.md) — design conventions and sticky decisions.
- [`docs/superpowers/specs/2026-05-12-cartesian-operators-design.md`](docs/superpowers/specs/2026-05-12-cartesian-operators-design.md)
  — original design rationale (historical).
