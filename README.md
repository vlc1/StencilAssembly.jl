# CartesianOperators

[![Build Status](https://github.com/vlc1/CartesianOperators.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/vlc1/CartesianOperators.jl/actions/workflows/CI.yml?query=branch%3Amain)

Julia package that assembles `SparseMatrixCSC` operators for stencil patterns
on rectangular Cartesian meshes. Row and column index sets are
`NTuple{N, AbstractUnitRange{Int}}`; stencil offsets are contiguous
unit-stride integers encoded as a `StaticArrays.SUnitRange{O, L}`. Sparsity
pattern and numerical fill are exposed as separate operations, both
allocation-free, so the fill can be re-run cheaply inside an outer
iterative solver.

```julia
using CartesianOperators, FillArrays
using StaticArrays: SUnitRange

row = (1:4,); col = (1:4,)

# Forward x-difference (D ϕ)[i] = ϕ[i+1] − ϕ[i].
# coefs are in ascending offset order: coefs[1] ↦ smallest offset.
forward = LinearStencil{1}(SUnitRange(0, 1), (Fill(-1.0, 4), Fill(1.0, 4)))

J = build(forward, row, col)         # one-shot assemble + update!
# Or split — pattern is built once and refilled cheaply across solver iterations:
J = assemble(forward, row, col)
update!(J, forward, row, col)
```

## Classical 1-D differences

| Operator   | constructor                                                                        | `(D ϕ)[i]`        |
|------------|------------------------------------------------------------------------------------|-------------------|
| Forward x  | `LinearStencil{1}(SUnitRange( 0, 1), (Fill(-1.0, n), Fill(1.0, n)))`               | `ϕ[i+1] − ϕ[i]`   |
| Backward x | `LinearStencil{1}(SUnitRange(-1, 0), (Fill(-1.0, n), Fill(1.0, n)))`               | `ϕ[i]   − ϕ[i-1]` |
| Central x  | `LinearStencil{1}(SUnitRange(-1, 1), (Fill(-1.0, n), Fill(0.0, n), Fill(1.0, n)))` | `ϕ[i+1] − ϕ[i-1]` |

Contiguity forces every offset between `Δ_min` and `Δ_max` to be represented,
so the central difference carries a structural zero at offset 0. For variable
coefficients pass any `AbstractArray` whose axes cover `col` (e.g.
`(-1 ./ ρ, 1 ./ ρ)` for a density-weighted gradient).

## Three operations

| Function                   | Allocates                            | Does                                            |
|----------------------------|--------------------------------------|-------------------------------------------------|
| `assemble(st, row, col)`   | `colptr` + `rowval` + uninit `nzval` | builds the sparsity pattern                     |
| `update!(J, st, row, col)` | no                                   | writes `J.nzval` in place                       |
| `build(st, row, col)`      | same as `assemble`                   | `update!(assemble(st, row, col), st, row, col)` |

`assemble` and `update!` pin `D = 1`, `N = 1` at the type level (misuse →
`MethodError`) and enforce **`L − 1 ≤ length(row[1])`** at runtime — the
three-phase kernel's exact correctness boundary.

## Status

Implemented: `LinearStencil{D, O, L, T, N, C}` (any `1 ≤ D ≤ N`, variable
coefficients) and 1-D `assemble` / `update!` / `build`. Next milestone: N-D
dispatch via a recursive dimensional-peeling kernel, then composition
(the Laplacian as a sum of `LinearStencil`s along each dimension).

See [`AGENTS.md`](AGENTS.md) for design decisions (type-driven API, row/col
contract, three-phase kernel shape, CSC ordering) and
[`docs/plan.md`](docs/plan.md) for the implementation plan. Historical
design rationale:
[`docs/superpowers/specs/2026-05-12-cartesian-operators-design.md`](docs/superpowers/specs/2026-05-12-cartesian-operators-design.md).
