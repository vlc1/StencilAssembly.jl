using SparseArrays

"""
    stencil_reference(offsets, coefs, row, col) -> SparseMatrixCSC

Brute-force reference. Enumerates every `c_idx ∈ CartesianIndices(col)` in mesh
space, applies each offset (`r_idx = c_idx - offsets[k]`), checks whether the
shifted-to-1-based row position lies in the row's compact range, and emits the
compact linear `(row, col, value)` triple.

Mesh-space `CartesianIndex`es are translated to 1-based local coords via fixed
shifts (`rsh`, `csh`) so we can use standard 1-based `LinearIndices` —
`LinearIndices((3:7,))` in Julia bounds-checks against `(OneTo(5),)`, not against
`(3:7,)`, so feeding shifted mesh positions to it directly is incorrect.

Used only by tests as a correctness oracle for the range-based kernels.
"""
function stencil_reference(
    offsets::NTuple{K,CartesianIndex{N}},
    coefs::NTuple{K,T},
    row::NTuple{N,AbstractUnitRange{Int}},
    col::NTuple{N,AbstractUnitRange{Int}},
) where {K,N,T}
    m, n = prod(length, row), prod(length, col)
    rsh = CartesianIndex(map(r -> 1 - first(r), row))   # mesh → 1-based, row side
    csh = CartesianIndex(map(r -> 1 - first(r), col))   # mesh → 1-based, col side
    Lr = LinearIndices(map(length, row))
    Lc = LinearIndices(map(length, col))
    Cr = CartesianIndices(map(length, row))
    I = Int[]; J = Int[]; V = T[]
    for c_idx in CartesianIndices(col)
        c_local = c_idx + csh
        for k in 1:K
            r_idx = c_idx - offsets[k]
            r_local = r_idx + rsh
            if r_local in Cr
                push!(I, Lr[r_local]); push!(J, Lc[c_local]); push!(V, coefs[k])
            end
        end
    end
    sparse(I, J, V, m, n)
end
