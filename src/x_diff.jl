"""
    _x_diff_pattern_runs!(rowval, colptr, offsets,
                          row_ivs, row_lo, row_hi,
                          col_ivs, col_lo, col_hi)

1-D pattern kernel for stencils whose offsets only act on dimension 1.

Walks `col_ivs[col_lo:col_hi]` in compact-ascending column order. For each
stencil offset Δ in `offsets`, maintains an independent monotone pointer into
`row_ivs[row_lo:row_hi]` and computes the shifted row mesh-position
`r = c - Δ` directly. Compact indices come from `Interval.shift` via range
arithmetic; no `Base.in` or `getindex` queries.

Appends row-compact indices to `rowval`. Sets `colptr[c_compact + 1] =
colptr[c_compact] + (entries emitted at column c_compact)`.

`offsets` must be sorted descending so per-column rowval is monotone ascending.
"""
function _x_diff_pattern_runs!(
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
    _x_diff_fill_runs!(nzval, colptr, offsets, coefs,
                       row_ivs, row_lo, row_hi,
                       col_ivs, col_lo, col_hi)

1-D fill kernel. Runs the **same** sweep as `_x_diff_pattern_runs!`, but
instead of appending to `rowval` it writes `nzval[colptr[c_compact] + slot] =
coefs[k]` where `slot` is the per-column running offset count and `k` the
index of the offset that hit. `offsets` and `coefs` must be in the same order.
Allocation-free apart from the K-element `rps` pointer buffer.
"""
function _x_diff_fill_runs!(
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
