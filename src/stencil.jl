"""
    LinearStencil{D,K,T,N,C<:NTuple{K,AbstractArray{T,N}}}

A variable-coefficient stencil aligned with mesh dimension `D`. For column `c`
at mesh position `p_c` and offset `Δ_k`, the matrix entry lands on the row at
mesh position `p_c − Δ_k` and carries coefficient `coefs[k][p_c]` — i.e., coefs
are anchored at the **column** mesh position (see `AGENTS.md`).

# Type parameters

- `D::Int` — mesh dimension on which the stencil acts (1-based).
- `K::Int` — number of stencil terms.
- `T` — shared element type of every coef array.
- `N` — coef-array dimensionality; matches the `CartesianRunIndices{N}` the
  stencil will be assembled against.
- `C` — concrete tuple type of the coef containers; inferred from the
  constructor call. Heterogeneous containers (e.g. `Fill` + `Vector`) are fine
  as long as they share `eltype` and `ndims`.

# Fields

- `offsets::NTuple{K,Int}` — strictly descending 1-D offsets along dim `D`.
- `coefs::C` — `NTuple{K,<:AbstractArray{T,N}}` of coefficient arrays.

# Construction

    LinearStencil{D}(offsets, coefs)

Inner constructor (well-typed path) validates `D ≥ 1` and strict-descending
offsets via `issorted(offsets; lt = >=)`. The shared `eltype`/`ndims` of
`coefs` are enforced by the method signature.

A catch-all outer constructor (`LinearStencil{D}(::Tuple, ::Tuple)`) reports
friendly errors when the inputs are ill-typed (length mismatch, non-`Int`
offsets, non-`AbstractArray` coefs, mixed `eltype`, or mixed `ndims`).

# Examples

```julia
using FillArrays
n = 5
forward_x = LinearStencil{1}((1, 0), (Fill(1.0, n), Fill(-1.0, n)))
# variable-coef: density-weighted gradient ψ[i] = (φ[i] − φ[i−1]) / ρ[i]
ρ = rand(n)
grad = LinearStencil{1}((0, -1), (1 ./ ρ, -1 ./ ρ))
```
"""
struct LinearStencil{D,K,T,N,C<:NTuple{K,AbstractArray{T,N}}}
    offsets::NTuple{K,Int}
    coefs::C

    function LinearStencil{D}(
        offsets::NTuple{K,Int},
        coefs::NTuple{K,AbstractArray{T,N}},
    ) where {D,K,T,N}
        D isa Int && D >= 1 || throw(ArgumentError(
            "stencil dimension D must be a positive Int (got $D)"))
        issorted(offsets; lt = >=) || throw(ArgumentError(
            "offsets must be strictly descending (got $offsets); required for SparseMatrixCSC rowval-per-column sortedness"))
        new{D,K,T,N,typeof(coefs)}(offsets, coefs)
    end
end

# Catch-all outer constructor: friendly errors for ill-typed inputs. Only fires
# when the inner method's NTuple{K,AbstractArray{T,N}} signature does not match
# (mixed eltype, mixed ndims, non-array elements, length mismatch, etc.).
function LinearStencil{D}(offsets::Tuple, coefs::Tuple) where {D}
    length(offsets) == length(coefs) || throw(ArgumentError(
        "offsets has length $(length(offsets)) but coefs has length $(length(coefs))"))
    all(o -> o isa Int, offsets) || throw(ArgumentError(
        "offsets must be a tuple of Int (got $(map(typeof, offsets)))"))
    all(c -> c isa AbstractArray, coefs) || throw(ArgumentError(
        "each coef must be an AbstractArray (got $(map(typeof, coefs)))"))
    Ts = map(eltype, coefs)
    all(==(first(Ts)), Ts) || throw(ArgumentError(
        "all coefs must share the same eltype (got $Ts)"))
    Ns = map(ndims, coefs)
    all(==(first(Ns)), Ns) || throw(ArgumentError(
        "all coefs must share the same ndims (got $Ns)"))
    throw(ArgumentError("LinearStencil could not be constructed; coefs = $coefs"))
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

1-D fill kernel. Runs the **same** sweep as `_pattern_runs!`, but writes
`nzval[colptr[c_compact] + slot] = coefs[k][c]` (column-anchored variable-coef
read) where `slot` is the per-column running offset count and `k` indexes the
offset that hit. `offsets` and `coefs` must be in the same order. Allocation-
free apart from the K-element `rps` pointer buffer (and whatever `getindex` on
the user's coef arrays costs — `Vector`, `Fill`, `OffsetArray` etc. are O(1)).
"""
function _fill_runs!(
    nzval::AbstractVector{T},
    colptr::Vector{Int},
    offsets::NTuple{K,Int},
    coefs::NTuple{K,AbstractArray{T,1}},
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
                        nzval[slot] = coefs[k][c]
                        slot += 1
                    end
                end
            end
        end
    end
    return
end

"""
    assemble(st::LinearStencil{D,K,T,1},
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
    st::LinearStencil{D,K,T,1},
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
    update!(mat::SparseMatrixCSC{T,Int}, st::LinearStencil{D,K,T,1},
            row::CartesianRunIndices{1}, col::CartesianRunIndices{1}) -> mat

Write `mat.nzval` in place by re-walking `row`/`col` with `st`. `mat` must
have been produced by a matching `assemble(st, row, col)` call (same stencil,
same cri's) so its `colptr`/`rowval` align with the sweep order.
Allocation-free up to a K-element scratch buffer.

Throws `ArgumentError` if `D ≠ 1`.
"""
function update!(
    mat::SparseMatrixCSC{T,Int},
    st::LinearStencil{D,K,T,1},
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
