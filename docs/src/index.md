# StencilAssembly.jl

StencilAssembly turns a **stencil** into a sparse matrix:
[`SparseMatrixCSC`](https://docs.julialang.org/en/v1/stdlib/SparseArrays/), the
default Julia format. It is the assembly half of a small stack — it depends on
[StencilCore](https://vlc1.github.io/StencilCore.jl/dev/) for the stencil types,
and pairs with [StencilCalculus](https://vlc1.github.io/StencilCalculus.jl/dev/),
which *produces* stencils by differentiating symbolic grid expressions.

## Why assembly is the interesting part

On a structured (Cartesian) mesh a discrete field is an N-D array, and a great
many operations — finite differences, Laplacians — are **stencils**: the same
local formula applied at every node. The matrix of such an operation (its
Jacobian, in a Newton solve) is therefore very **sparse** and highly
structured: every column has the same handful of nonzeros, at fixed diagonal
offsets. The job of this package is to walk that structure once and fill the
three arrays of a CSC matrix — `colptr`, `rowval`, `nzval` — without waste.

Two ideas keep it efficient and N-D-uniform:

- **Index spaces are ranges.** The row and column index sets are
  `NTuple{N, AbstractUnitRange{Int}}` — a rectangular sub-block of the mesh per
  space. Julia's `LinearIndices` / `CartesianIndices` convert between the
  Cartesian and linear numbering, and the kernels visit each output column
  exactly once by **peeling dimensions** recursively (outermost to innermost).
- **Coefficients are column-anchored.** Offsets are *diagonal indices*
  (`δ = column − row`), and `term[c][k]` is the coefficient at **column** `c`
  for offset `k` — exactly the order CSC wants, so rows come out sorted with no
  post-sort. (This is the `ColumnAccess` style; `RowAccess` is reserved for a
  future CSR backend.)

## Three operations

| Function | allocates | does |
|---|---|---|
| [`assemble`](@ref)`(st, row, col)` | `colptr` + `rowval` + uninit `nzval` | the sparsity pattern |
| [`update!`](@ref)`(A, st, row, col)` | nothing | writes `A.nzval` in place |
| [`build`](@ref)`(st, row, col)` | as `assemble` | both at once |

Splitting pattern from fill lets the (allocation-free) `update!` be re-run
cheaply across the iterations of an outer solver, reusing the sparsity pattern.

## Quickstart

```julia
using StencilAssembly, StaticArrays

n = 6
# Forward x-difference: offsets 0,1; coefficients (-1, 1) on every column.
fwd = LinearStencil{1}(SUnitRange(0, 1), fill(SVector(-1.0, 1.0), n))
A = build(fwd, (1:n,), (1:n,))      # SparseMatrixCSC; (A*f)[i] = f[i+1] - f[i]
```

A stencil whose coefficient is still *symbolic* (an `AbstractTerm`) is not
assemblable — `assemble` it only after
[StencilCalculus](https://vlc1.github.io/StencilCalculus.jl/dev/) has
`materialize`d it to a concrete array. See the [Guide](@ref) for the 2-D
Laplacian and variable-coefficient cases, and the [API reference](@ref).
