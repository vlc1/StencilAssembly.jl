# naive implementation works only when column and row interval sets share the
# same underlying integer mesh (`dom` argument).

using CartesianRuns
using SparseArrays
using Random

const CRI = CartesianRunIndices

function stencil_naive_x(
    dom::NTuple{N,<:AbstractUnitRange{Int}},
    offsets::NTuple{K,Int},
    coefs::NTuple{K,T},
    row::CRI{N},
    col::CRI{N},
) where {N,K,T}
    n = prod(length, dom)

    diags = map(offsets, coefs) do offset, coef
        Pair(offset, coef * ones(T, max(n - abs(offset), 0)))
    end

    full = spdiagm(diags...)

    row_ = getindex(LinearIndices(dom), row)
    col_ = getindex(LinearIndices(dom), col)

    full[row_, col_]
end

include("reference.jl")

Random.seed!(0)
dom = (1:16,)
row = CartesianRunIndices(rand(Bool, 16))
col = CartesianRunIndices(rand(Bool, 16))

# forward_x
F1 = stencil_naive_x(dom, (1, 0), (1.0, -1.0), row, col)
F2 = stencil_reference((CartesianIndex(1), CartesianIndex(0)), (1.0, -1.0), row, col)

@show F1 == F2

# backward_x
B1 = stencil_naive_x(dom, (0, -1), (1.0, -1.0), row, col)
B2 = stencil_reference((CartesianIndex(0), CartesianIndex(-1)), (1.0, -1.0), row, col)

@show B1 == B2

# central_x
C1 = stencil_naive_x(dom, (1, -1), (1.0, -1.0), row, col)
C2 = stencil_reference((CartesianIndex(1), CartesianIndex(-1)), (1.0, -1.0), row, col)

@show C1 == C2

nothing
