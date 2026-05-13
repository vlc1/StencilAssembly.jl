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

J = forward_x_pattern(row, col)   # SparseMatrixCSC{Float64,Int}, nzval undef
forward_x_fill!(J, row, col)      # writes J.nzval allocation-free
```

## How operations work

### Two-stage assembly

Every operator is exposed as a pair:

| Function | Allocates? | Does what |
|---|---|---|
| `*_pattern(row_cri, col_cri, T = Float64)` | yes | builds `colptr` and `rowval`; allocates `nzval` but leaves it undef |
| `*_fill!(J, row_cri, col_cri)` | no (apart from a `K`-element buffer where `K = #offsets`) | writes `J.nzval` in place |

Symbolic structure is computed once per masked stencil; numeric fill is reused
many times in an outer non-linear solve.

### Stencils and the subtraction convention

A stencil is a small set of mesh-space `CartesianIndex` offsets. For each
offset `Δ`, the matrix entry from column `c` (at mesh position `p_c`) lands on
the row at mesh position `p_c − Δ`. Equivalently, row `r` (at mesh `p_r`)
contributes at column `p_r + Δ`. This is the **subtraction convention**: the
offset goes *from row to column*. It matches the mathematical reading of
`(D phi)[i] = … phi[i+Δ] …`, where the row index is `i` and the column index
is `i + Δ`.

Concretely, the three first-order x-differences ship with these offsets:

| Operator | Offsets (descending) | Coefs | Stencil |
|---|---|---|---|
| `forward_x_*`  | `(1, 0)`  | `(1, -1)` | `(D phi)[i] = phi[i+1] - phi[i]`  |
| `backward_x_*` | `(0, -1)` | `(1, -1)` | `(D phi)[i] = phi[i]   - phi[i-1]`|
| `central_x_*`  | `(1, -1)` | `(1, -1)` | `(D phi)[i] = phi[i+1] - phi[i-1]`|

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

`row_cri` and `col_cri` are interpreted on a shared integer mesh. Nothing in
the kernel or wrappers reads `domain(row_cri)` or `domain(col_cri)` — the
matrix structure depends only on the cri's intervals and the offsets. The
caller is responsible for ensuring the mesh interpretation is coherent
(typically: both cri's come from masks of the same physical grid). If the
cri's were constructed from masks on truly disjoint domains, the resulting
matrix is simply empty (no stencil offset bridges the two), which is
mathematically correct.

This is a deliberate choice: it lets you build operators between two cri's
that don't necessarily share an underlying boolean-mask domain (e.g., two
different masks on the same logical mesh, or one mask embedded in another).

### Offset ordering is tied to `SparseMatrixCSC`

`SparseMatrixCSC` requires `rowval` to be sorted ascending within each column.
We satisfy that **without an explicit per-column sort** by walking stencil
offsets in a fixed order. Under the subtraction convention, for column `c`
and offset `Δ`, the row mesh-position is `c − Δ`. Iterating offsets
**descending** (in column-major lexicographic order for N-D) makes the rows
emitted per column come out monotonically ascending in compact space — free.

This requirement is intrinsic to CSC. If the target format were CSR (per-row
storage with `colind` sorted within row), the comparator would flip to
ascending. For COO with deferred sort the discipline disappears, at the cost
of an O(`nnz`) sort step.

At the public API boundary the wrappers call
`_check_offsets_sorted_descending(offsets)`, which throws `ArgumentError` if
the constraint is violated. With hard-coded operator wrappers the check is a
no-op; it becomes load-bearing once the API accepts user-supplied stencils.

## Status

This is a work in progress. Currently implemented:

- 1-D forward, backward, and central x-differences (`forward_x_*`,
  `backward_x_*`, `central_x_*`).

Planned (Phase 1b onward): N-D recursive kernel via dimensional peeling,
y-direction differences, Laplacian, then generalization to a `Stencil` API.
See `docs/superpowers/specs/2026-05-12-cartesian-operators-design.md` for the
design rationale and `docs/superpowers/plans/2026-05-12-phase0-phase1.md` for
the implementation plan.
