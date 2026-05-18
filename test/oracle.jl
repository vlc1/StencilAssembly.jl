# Small cross-check script (not part of `Pkg.test()`): exercises `assemble` +
# `update!` against `stencil_reference` on a 1-D range, for forward, backward,
# and central differences. Run manually from the package root with:
#
#     julia --project=. test/oracle.jl

using CartesianOperators
using FillArrays
using SparseArrays
using StaticArrays: SUnitRange

include("reference.jl")

row = (1:16,)
col = (1:16,)
n = length(col[1])

# Coefs are in ascending offset order. Central difference is widened to
# SUnitRange(-1, 1) with an explicit zero middle (matches what the oracle
# emits when fed the same offsets).
st_f = LinearStencil{1}(SUnitRange( 0, 1), (Fill(-1.0, n), Fill(1.0, n)))
st_b = LinearStencil{1}(SUnitRange(-1, 0), (Fill(-1.0, n), Fill(1.0, n)))
st_c = LinearStencil{1}(SUnitRange(-1, 1), (Fill(-1.0, n), Fill(0.0, n), Fill(1.0, n)))

F = assemble(st_f, row, col); update!(F, st_f, row, col)
B = assemble(st_b, row, col); update!(B, st_b, row, col)
C = assemble(st_c, row, col); update!(C, st_c, row, col)

Fr = stencil_reference((CartesianIndex( 0), CartesianIndex(1)),                    (-1.0, 1.0),      row, col)
Br = stencil_reference((CartesianIndex(-1), CartesianIndex(0)),                    (-1.0, 1.0),      row, col)
Cr = stencil_reference((CartesianIndex(-1), CartesianIndex(0), CartesianIndex(1)), (-1.0, 0.0, 1.0), row, col)

@show F == Fr
@show B == Br
@show C == Cr

nothing
