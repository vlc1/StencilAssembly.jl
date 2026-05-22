# StencilAssembly

[![Build Status](https://github.com/vlc1/StencilAssembly.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/vlc1/StencilAssembly.jl/actions/workflows/CI.yml?query=branch%3Amain)

Julia package that assembles `SparseMatrixCSC` operators for stencil patterns
on rectangular Cartesian meshes. Row and column index sets are
`NTuple{N, AbstractUnitRange{Int}}`; the offset of a term is its
**diagonal index** in the numerical-linear-algebra sense — for column `j`
and row `i`, the diagonal is `k = j − i`. Sparsity pattern and numerical
fill are exposed as separate operations, both allocation-free, so the
fill can be re-run cheaply inside an outer iterative solver.

The stencil **types** live in [StencilCore](../StencilCore) (shared with the
symbolic CAS [StencilCalculus](../StencilCalculus)); this package depends on it and
provides the CSC **assembly**. Clone the three repos side by side — they
resolve each other through relative `[sources]` paths.

```julia
using StencilAssembly, FillArrays
using StaticArrays: SUnitRange, SVector

row = (1:4,); col = (1:4,)

# Forward x-difference (D ϕ)[i] = ϕ[i+1] − ϕ[i].
# The coefficient array holds one SVector per column, in ascending offset
# order: term[c][1] ↦ smallest offset.
forward = LinearStencil{1}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 4))

J = build(forward, row, col)         # one-shot assemble + update!
# Or split — pattern is built once and refilled cheaply across solver iterations:
J = assemble(forward, row, col)
update!(J, forward, row, col)
```

## Classical 1-D differences

| Operator   | constructor                                                          | `(D ϕ)[i]`        |
|------------|---------------------------------------------------------------------|-------------------|
| Forward x  | `LinearStencil{1}(SUnitRange( 0, 1), Fill(SVector(-1.0, 1.0), n))`      | `ϕ[i+1] − ϕ[i]`   |
| Backward x | `LinearStencil{1}(SUnitRange(-1, 0), Fill(SVector(-1.0, 1.0), n))`      | `ϕ[i]   − ϕ[i-1]` |
| Central x  | `LinearStencil{1}(SUnitRange(-1, 1), Fill(SVector(-1.0, 0.0, 1.0), n))` | `ϕ[i+1] − ϕ[i-1]` |

Contiguity forces every offset between `δ_min` and `δ_max` to be represented,
so the central difference carries a structural zero at offset 0. For variable
coefficients pass any `AbstractArray{<:SVector{L}}` whose axes cover `col`
(e.g. `SVector.(-1 ./ ρ, 1 ./ ρ)` for a density-weighted gradient). Because a
column's whole `SVector` is fetched in one `getindex`, a lazy coefficient
array can precompute quantities shared across that column's offsets.

## N-D star-shaped operators

`StarStencil{L}` is an N-D star with symmetric reach `−L … +L` per axis,
stored **interlaced**: one `SVector{M}` per cell (`M = 2NL + 1`) holding the
whole star in reverse-lex offset order, with the diagonal as the explicit
middle slot. Unlike a per-axis decomposition, the diagonal is a *free*
coefficient — so Helmholtz (`k²`) and parabolic (`∂ₜ`) terms have a home.

```julia
using StencilAssembly
using StaticArrays: SVector

n1, n2 = 5, 4
# 2-D negative Laplacian on a 5×4 mesh (L = 1 ⇒ M = 2·2·1 + 1 = 5).
# Reverse-lex order: (axis2,−1), (axis1,−1), diagonal, (axis1,+1), (axis2,+1).
coef = fill(SVector(-1.0, -1.0, 4.0, -1.0, -1.0), n1, n2)
lap  = StarStencil{1}(coef)
J = build(lap, (1:n1, 1:n2), (1:n1, 1:n2))
```

The diagonal slot is set directly (here `4`); the kernel walks each column's
`SVector` in reverse-lex (CSC) order and carries a per-axis guard
`2L ≤ length(row[d])`.

## Three operations

| Function                   | Allocates                            | Does                                            |
|----------------------------|--------------------------------------|-------------------------------------------------|
| `assemble(st, row, col)`   | `colptr` + `rowval` + uninit `nzval` | builds the sparsity pattern                     |
| `update!(J, st, row, col)` | no                                   | writes `J.nzval` in place                       |
| `build(st, row, col)`      | same as `assemble`                   | `update!(assemble(st, row, col), st, row, col)` |

For 1-D `LinearStencil` dispatch pins `D = 1` and `N = 1`; the runtime
guard is **`L − 1 ≤ length(row[1])`** (the three-phase kernel's exact
correctness boundary). N-D `LinearStencil` and `StarStencil` carry
analogous per-axis guards.

## Access style — CSC today, CSR tomorrow

Every concrete stencil subtypes `AbstractStencil{S<:AccessStyle}`, where
`S` is the **last** type parameter. `S` is a Holy-trait tag reporting how
the term arrays are anchored:

- `ColumnAccess` (default): `term[c_idx...][k]` reads at column mesh
  position `c_idx`. **Required for CSC** (`SparseMatrixCSC`).
- `RowAccess`: `term[r_idx...][k]` reads at row mesh position `r_idx`.
  Reserved for a future CSR assembler.

Default constructors return `ColumnAccess` stencils. Use a positional
`Type` tag to override:

```julia
csc_st = LinearStencil{1}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 5))
csr_st = LinearStencil{1}(RowAccess, SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 5))

AccessStyle(csc_st) === ColumnAccess()
AccessStyle(csr_st) === RowAccess()
```

`assemble` / `update!` are defined only for `ColumnAccess`; routing a
`RowAccess` stencil to the current CSC path raises a `MethodError`. The
trait is otherwise inert at runtime — it doesn't add overhead inside
the kernels.

## Status

Implemented: CSC assembly for `LinearStencil` (any `1 ≤ D ≤ N`) and the
interlaced `StarStencil` (any `N ≥ 1`), default `ColumnAccess`; 1-D, 2-D, 3-D
coverage against a brute-force oracle, plus the `Stencil`-narrowing path from
[StencilCalculus](../StencilCalculus). Next: a CSR assembler activating the `RowAccess`
path, then stencil composition.

See [`AGENTS.md`](AGENTS.md) for the CSC assembly invariants and
[`../StencilCore/AGENTS.md`](../StencilCore/AGENTS.md) for the type vocabulary.
The symbolic-CAS design is [`docs/cas.md`](../StencilCalculus/docs/cas.md); the package-split
design is [`docs/core.md`](../StencilCore/docs/core.md). (`docs/plan.md` / `docs/star.md` /
`docs/term.md` are earlier per-feature plans, partly superseded.)
