"""
    _isless_columnmajor(a::CartesianIndex{N}, b::CartesianIndex{N}) -> Bool

Strict column-major lexicographic comparison: `a < b` iff `a[N] < b[N]`, or
`a[N] == b[N]` and the dim-`(N-1)..1` components compare strictly less. The
outermost dimension is most significant, matching the natural compact order of
`CartesianRunIndices`.
"""
@inline function _isless_columnmajor(
    a::CartesianIndex{N}, b::CartesianIndex{N},
) where {N}
    ta = Tuple(a); tb = Tuple(b)
    for d in N:-1:1
        ta[d] < tb[d] && return true
        ta[d] > tb[d] && return false
    end
    return false
end

"""
    _check_offsets_sorted_descending(offsets)

Throw `ArgumentError` unless `offsets::NTuple{K,CartesianIndex{N}}` is strictly
descending in column-major lex. Required so that the pointer-sweep kernels emit
per-column `rowval` monotonically ascending — the invariant `SparseMatrixCSC`
demands. See "How operations work" in the README for the tie between offset
ordering and the CSC representation.
"""
function _check_offsets_sorted_descending(
    offsets::NTuple{K,CartesianIndex{N}},
) where {K,N}
    for k in 2:K
        _isless_columnmajor(offsets[k], offsets[k - 1]) || throw(ArgumentError(
            "offsets must be strictly descending in column-major lex (got $offsets); required for SparseMatrixCSC rowval-per-column sortedness"))
    end
    return nothing
end

"""
    _dim1_int_offsets(offsets::NTuple{K,CartesianIndex{1}}) -> NTuple{K,Int}

Extract the dim-1 integer components of `offsets`. Used at the boundary
between the public-ish `(offsets::NTuple{K,CartesianIndex{1}}, coefs)` layer
and the `NTuple{K,Int}`-taking pointer-sweep kernels.
"""
@inline _dim1_int_offsets(offsets::NTuple{K,CartesianIndex{1}}) where {K} =
    map(Δ -> Tuple(Δ)[1], offsets)

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

# ── Intermediate 1-D-aligned-with-x layer ─────────────────────────────────────
#
# These two functions parameterise the operator wrappers by `(offsets, coefs)`
# tuples, mirroring the calling convention of `stencil_reference`
# (`test/reference.jl`) and `stencil_naive_x` (`test/oracle.jl`). The hard-
# coded `forward_x_*`, `backward_x_*`, `central_x_*` wrappers below delegate
# to these.
#
# Phase 1b will add `where {N ≥ 2}` companions that dispatch to a recursive
# fused kernel; the 1-D dispatch here will remain unchanged.

"""
    x_diff_pattern_1d(offsets, row_cri, col_cri, T = Float64) -> SparseMatrixCSC{T,Int}

Sparsity pattern of a stencil operator whose offsets act on dimension 1 only,
applied to a 1-D row/column masked index set.

- `offsets::NTuple{K,CartesianIndex{1}}` — stencil offsets, **strictly
  descending** in column-major lex (in 1-D, just strictly descending integer).
- `row_cri`, `col_cri` — 1-D `CartesianRunIndices` on a shared integer mesh.
- `T` — element type of the returned matrix (`nzval` is allocated but undef).

Throws `ArgumentError` if `offsets` is not strictly descending.
"""
function x_diff_pattern_1d(
    offsets::NTuple{K,CartesianIndex{1}},
    row_cri::CartesianRunIndices{1}, col_cri::CartesianRunIndices{1},
    ::Type{T} = Float64,
) where {K,T}
    _check_offsets_sorted_descending(offsets)
    offs_int = _dim1_int_offsets(offsets)
    m, n = length(row_cri), length(col_cri)
    colptr = Vector{Int}(undef, n + 1); colptr[1] = 1
    rowval = Int[]
    _x_diff_pattern_runs!(rowval, colptr, offs_int,
        row_cri.intervals[1], 1, length(row_cri.intervals[1]),
        col_cri.intervals[1], 1, length(col_cri.intervals[1]))
    nzval = Vector{T}(undef, length(rowval))
    SparseMatrixCSC{T,Int}(m, n, colptr, rowval, nzval)
end

"""
    x_diff_fill_1d!(J, offsets, coefs, row_cri, col_cri) -> J

Fill `J.nzval` for a 1-D-aligned-with-x stencil. `J` must have been built by
`x_diff_pattern_1d(offsets, row_cri, col_cri, eltype(J))` (or by one of the
hard-coded `*_pattern` wrappers that delegates to it). `offsets` and `coefs`
must be in the same order; both ordered strictly descending in column-major lex
(see `x_diff_pattern_1d`).
"""
function x_diff_fill_1d!(
    J::SparseMatrixCSC{T,Int},
    offsets::NTuple{K,CartesianIndex{1}},
    coefs::NTuple{K,T},
    row_cri::CartesianRunIndices{1}, col_cri::CartesianRunIndices{1},
) where {K,T}
    _check_offsets_sorted_descending(offsets)
    offs_int = _dim1_int_offsets(offsets)
    _x_diff_fill_runs!(J.nzval, J.colptr, offs_int, coefs,
        row_cri.intervals[1], 1, length(row_cri.intervals[1]),
        col_cri.intervals[1], 1, length(col_cri.intervals[1]))
    return J
end

# ── Hard-coded operator wrappers ──────────────────────────────────────────────

const _FORWARD_X_OFFSETS  = (CartesianIndex(1),  CartesianIndex(0))
const _BACKWARD_X_OFFSETS = (CartesianIndex(0),  CartesianIndex(-1))
const _CENTRAL_X_OFFSETS  = (CartesianIndex(1),  CartesianIndex(-1))

"""
    forward_x_pattern(row_cri, col_cri, T = Float64) -> SparseMatrixCSC{T,Int}

Sparsity pattern of the forward x-difference operator
`(D phi)[i] = phi[i+1] - phi[i]` in 1-D.
"""
forward_x_pattern(
    row_cri::CartesianRunIndices{1}, col_cri::CartesianRunIndices{1},
    ::Type{T} = Float64,
) where {T} = x_diff_pattern_1d(_FORWARD_X_OFFSETS, row_cri, col_cri, T)

"""
    forward_x_fill!(J, row_cri, col_cri) -> J

Fill `J.nzval` for the forward x-difference operator.
"""
function forward_x_fill!(
    J::SparseMatrixCSC{T,Int},
    row_cri::CartesianRunIndices{1}, col_cri::CartesianRunIndices{1},
) where {T}
    x_diff_fill_1d!(J, _FORWARD_X_OFFSETS, (T(1), T(-1)), row_cri, col_cri)
end

"""
    backward_x_pattern(row_cri, col_cri, T = Float64) -> SparseMatrixCSC{T,Int}

Sparsity pattern of the backward x-difference operator
`(D phi)[i] = phi[i] - phi[i-1]` in 1-D.
"""
backward_x_pattern(
    row_cri::CartesianRunIndices{1}, col_cri::CartesianRunIndices{1},
    ::Type{T} = Float64,
) where {T} = x_diff_pattern_1d(_BACKWARD_X_OFFSETS, row_cri, col_cri, T)

"""
    backward_x_fill!(J, row_cri, col_cri) -> J
"""
function backward_x_fill!(
    J::SparseMatrixCSC{T,Int},
    row_cri::CartesianRunIndices{1}, col_cri::CartesianRunIndices{1},
) where {T}
    x_diff_fill_1d!(J, _BACKWARD_X_OFFSETS, (T(1), T(-1)), row_cri, col_cri)
end

"""
    central_x_pattern(row_cri, col_cri, T = Float64) -> SparseMatrixCSC{T,Int}

Sparsity pattern of the central x-difference operator
`(D phi)[i] = phi[i+1] - phi[i-1]` in 1-D.
"""
central_x_pattern(
    row_cri::CartesianRunIndices{1}, col_cri::CartesianRunIndices{1},
    ::Type{T} = Float64,
) where {T} = x_diff_pattern_1d(_CENTRAL_X_OFFSETS, row_cri, col_cri, T)

"""
    central_x_fill!(J, row_cri, col_cri) -> J
"""
function central_x_fill!(
    J::SparseMatrixCSC{T,Int},
    row_cri::CartesianRunIndices{1}, col_cri::CartesianRunIndices{1},
) where {T}
    x_diff_fill_1d!(J, _CENTRAL_X_OFFSETS, (T(1), T(-1)), row_cri, col_cri)
end
