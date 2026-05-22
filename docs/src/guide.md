# Guide

Worked examples of assembling stencils into `SparseMatrixCSC`. Stencils here are
built by hand; in practice you would obtain them from
[StencilCalculus](https://vlc1.github.io/StencilCalculus.jl/dev/).

## Forward difference (1-D)

```julia
using StencilAssembly, StaticArrays

n = 6
fwd = LinearStencil{1}(SUnitRange(0, 1), fill(SVector(-1.0, 1.0), n))
A = build(fwd, (1:n,), (1:n,))
# (A * f)[i] == f[i+1] - f[i]  for i = 1 … n-1
```

## Pattern / fill split

`build` is `update!(assemble(...), ...)`. Keep the pattern and refill across
solver iterations — `update!` is allocation-free:

```julia
A = assemble(fwd, (1:n,), (1:n,))   # colptr + rowval + uninitialised nzval
update!(A, fwd, (1:n,), (1:n,))     # writes nzval in place; repeatable
```

## Variable coefficients

Coefficients are arbitrary arrays, indexed at the **column**. A density-weighted
gradient `ψ[i] = (ϕ[i] − ϕ[i−1]) / ρ[i]` has offsets `-1, 0` and per-column
coefficients `(−1/ρ, 1/ρ)`:

```julia
ρ = [2.0, 3.0, 5.0, 7.0]; n = 4
# element[1] ↦ offset -1, element[2] ↦ offset 0 (ascending-offset order)
term = SVector.(vcat(-1 ./ ρ[2:end], 0.0), 1 ./ ρ)
grad = LinearStencil{1}(SUnitRange(-1, 0), term)
A = build(grad, (1:n,), (1:n,))
```

Because a whole column's coefficients are fetched in one `getindex`, a lazy
coefficient array can precompute quantities shared across a column's offsets.

## 2-D Laplacian (StarStencil)

The interlaced [`StarStencil`](@ref) packs the whole star into one `SVector` per
cell (reverse-lex order, diagonal in the middle slot):

```julia
n1, n2 = 5, 4
# five-point Laplacian: off-diagonals -1, diagonal +4.
lap = StarStencil{1}(fill(SVector(-1.0, -1.0, 4.0, -1.0, -1.0), n1, n2))
A = build(lap, (1:n1, 1:n2), (1:n1, 1:n2))
```

Unequal or shifted `row`/`col` ranges are allowed (rectangular sub-blocks on a
shared mesh); the coefficient array's axes must cover `col`. The kernel carries
a per-axis guard `2L ≤ length(row[d])`.
