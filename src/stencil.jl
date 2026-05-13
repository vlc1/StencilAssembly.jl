"""
    LinearStencil{D,K,T}

A constant-coefficient stencil aligned with mesh dimension `D`. The matrix
entry from column `c` (at mesh position `p_c`) to row `r = p_c − Δ_k` carries
coefficient `coefs[k]`, for each `k` in `1:K`.

# Type parameters

- `D::Int` — mesh dimension on which the stencil acts (1-based).
- `K::Int` — number of stencil terms.
- `T` — coefficient eltype.

# Fields

- `offsets::NTuple{K,Int}` — strictly descending 1-D offsets along dim `D`.
- `coefs::NTuple{K,T}` — matching coefficients in the same order.

# Construction

    LinearStencil{D}(offsets, coefs)

Strict-descending offset order is required by the `SparseMatrixCSC` `rowval`-
per-column sortedness invariant; the constructor validates via
`issorted(offsets; lt = >=)`. `D ≥ 1` is also validated.

# Examples

```julia
forward_x  = LinearStencil{1}((1,  0), (1.0, -1.0))   # (D ϕ)[i] = ϕ[i+1] − ϕ[i]
backward_x = LinearStencil{1}((0, -1), (1.0, -1.0))   # (D ϕ)[i] = ϕ[i]   − ϕ[i-1]
central_x  = LinearStencil{1}((1, -1), (1.0, -1.0))   # (D ϕ)[i] = ϕ[i+1] − ϕ[i-1]
```
"""
struct LinearStencil{D,K,T}
    offsets::NTuple{K,Int}
    coefs::NTuple{K,T}

    function LinearStencil{D}(
        offsets::NTuple{K,Int},
        coefs::NTuple{K,T},
    ) where {D,K,T}
        D isa Int && D >= 1 || throw(ArgumentError(
            "stencil dimension D must be a positive Int (got $D)"))
        issorted(offsets; lt = >=) || throw(ArgumentError(
            "offsets must be strictly descending (got $offsets); required for SparseMatrixCSC rowval-per-column sortedness"))
        new{D,K,T}(offsets, coefs)
    end
end

"""
    _pattern_runs!(rowval, colptr, offsets,
                   row_ivs, row_lo, row_hi,
                   col_ivs, col_lo, col_hi)

1-D pattern kernel. Walks `col_ivs[col_lo:col_hi]` in compact-ascending column
order; for each offset Δ in `offsets`, maintains an independent monotone
pointer into `row_ivs[row_lo:row_hi]` and computes the shifted row mesh
position `r = c − Δ` directly. Compact indices come from `Interval.shift` via
range arithmetic; no `Base.in` or `getindex` queries.

Appends row-compact indices to `rowval`. Sets `colptr[c_compact + 1] =
colptr[c_compact] + (entries emitted at column c_compact)`.

`offsets` must be sorted strictly descending so per-column `rowval` is
monotone ascending — the `SparseMatrixCSC` invariant.
"""
function _pattern_runs!(
    rowval::Vector{Int},
    colptr::Vector{Int},
    offsets::NTuple{K,Int},
    row_ivs::AbstractVector{Interval}, row_lo::Int, row_hi::Int,
    col_ivs::AbstractVector{Interval}, col_lo::Int, col_hi::Int,
) where {K}
    rps = fill(row_lo, K)
    for k_iv in col_lo:col_hi
        col_iv = col_ivs[k_iv]
        col_sh = shift(col_iv)
        for c in col_iv.mask
            c_compact = c + col_sh
            cnt = 0
            for k in 1:K
                Δ = offsets[k]
                r = c - Δ
                while rps[k] <= row_hi && row_ivs[rps[k]].mask.stop < r
                    rps[k] += 1
                end
                if rps[k] <= row_hi
                    row_iv = row_ivs[rps[k]]
                    if row_iv.mask.start <= r
                        push!(rowval, r + shift(row_iv))
                        cnt += 1
                    end
                end
            end
            colptr[c_compact + 1] = colptr[c_compact] + cnt
        end
    end
    return
end

"""
    _fill_runs!(nzval, colptr, offsets, coefs,
                row_ivs, row_lo, row_hi,
                col_ivs, col_lo, col_hi)

1-D fill kernel. Runs the **same** sweep as `_pattern_runs!`, but instead of
appending to `rowval` it writes `nzval[colptr[c_compact] + slot] = coefs[k]`
where `slot` is the per-column running offset count and `k` is the index of
the offset that hit. `offsets` and `coefs` must be in the same order.
Allocation-free apart from the K-element `rps` pointer buffer.
"""
function _fill_runs!(
    nzval::AbstractVector{T},
    colptr::Vector{Int},
    offsets::NTuple{K,Int},
    coefs::NTuple{K,T},
    row_ivs::AbstractVector{Interval}, row_lo::Int, row_hi::Int,
    col_ivs::AbstractVector{Interval}, col_lo::Int, col_hi::Int,
) where {K,T}
    rps = fill(row_lo, K)
    for k_iv in col_lo:col_hi
        col_iv = col_ivs[k_iv]
        col_sh = shift(col_iv)
        for c in col_iv.mask
            c_compact = c + col_sh
            slot = colptr[c_compact]
            for k in 1:K
                Δ = offsets[k]
                r = c - Δ
                while rps[k] <= row_hi && row_ivs[rps[k]].mask.stop < r
                    rps[k] += 1
                end
                if rps[k] <= row_hi
                    row_iv = row_ivs[rps[k]]
                    if row_iv.mask.start <= r
                        nzval[slot] = coefs[k]
                        slot += 1
                    end
                end
            end
        end
    end
    return
end

"""
    assemble(st::LinearStencil{D,K,T},
             row::CartesianRunIndices{1}, col::CartesianRunIndices{1},
             ::Type{SparseMatrixCSC{T,Int}} = SparseMatrixCSC{T,Int})
        -> SparseMatrixCSC{T,Int}

Build the sparsity pattern (`colptr`, `rowval`) of the operator induced by
`st` between the masked row and column index sets, and allocate `nzval`
**uninitialised**. Call `update!` to populate `nzval` before using the matrix.

The trailing `::Type` argument selects the matrix target; v1 supports only
`SparseMatrixCSC{T,Int}` and the parameter is positional for future extension
to other matrix formats.

Throws `ArgumentError` if `D ≠ 1` (1-D dispatch only; N-D follows in a
subsequent step).
"""
function assemble(
    st::LinearStencil{D,K,T},
    row::CartesianRunIndices{1},
    col::CartesianRunIndices{1},
    ::Type{SparseMatrixCSC{T,Int}} = SparseMatrixCSC{T,Int},
) where {D,K,T}
    D == 1 || throw(ArgumentError(
        "1-D CartesianRunIndices requires LinearStencil{1}; got LinearStencil{$D}"))
    m, n = length(row), length(col)
    colptr = Vector{Int}(undef, n + 1); colptr[1] = 1
    rowval = Int[]
    _pattern_runs!(rowval, colptr, st.offsets,
        row.intervals[1], 1, length(row.intervals[1]),
        col.intervals[1], 1, length(col.intervals[1]))
    nzval = Vector{T}(undef, length(rowval))
    SparseMatrixCSC{T,Int}(m, n, colptr, rowval, nzval)
end

"""
    update!(mat::SparseMatrixCSC{T,Int}, st::LinearStencil{D,K,T},
            row::CartesianRunIndices{1}, col::CartesianRunIndices{1}) -> mat

Write `mat.nzval` in place by re-walking `row`/`col` with `st`. `mat` must
have been produced by a matching `assemble(st, row, col)` call (same stencil,
same cri's) so its `colptr`/`rowval` align with the sweep order.
Allocation-free up to a K-element scratch buffer.

Throws `ArgumentError` if `D ≠ 1`.
"""
function update!(
    mat::SparseMatrixCSC{T,Int},
    st::LinearStencil{D,K,T},
    row::CartesianRunIndices{1},
    col::CartesianRunIndices{1},
) where {D,K,T}
    D == 1 || throw(ArgumentError(
        "1-D CartesianRunIndices requires LinearStencil{1}; got LinearStencil{$D}"))
    _fill_runs!(mat.nzval, mat.colptr, st.offsets, st.coefs,
        row.intervals[1], 1, length(row.intervals[1]),
        col.intervals[1], 1, length(col.intervals[1]))
    return mat
end
