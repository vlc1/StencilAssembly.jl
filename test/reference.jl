using CartesianRuns: CartesianRunIndices
using SparseArrays

"""
    stencil_reference(offsets, coefs, row_cri, col_cri) -> SparseMatrixCSC

Brute-force reference: walks `col_cri` cell-by-cell, applies each offset, looks
up the row-compact index via a Dict, and emits (row, col, val) triples. Used
only by tests as a correctness oracle for the pointer-sweep kernels.
"""
function stencil_reference(
    offsets::NTuple{K,CartesianIndex{N}},
    coefs::NTuple{K,T},
    row_cri::CartesianRunIndices{N},
    col_cri::CartesianRunIndices{N},
) where {K,N,T}
    m, n = length(row_cri), length(col_cri)
    row_lookup = Dict{CartesianIndex{N},Int}()
    for (i, idx) in enumerate(row_cri)
        row_lookup[idx] = i
    end
    I = Int[]; J = Int[]; V = T[]
    for (j, c_idx) in enumerate(col_cri)
        for k in 1:K
            r_idx = c_idx + offsets[k]
            i = get(row_lookup, r_idx, 0)
            if i != 0
                push!(I, i); push!(J, j); push!(V, coefs[k])
            end
        end
    end
    sparse(I, J, V, m, n)
end
