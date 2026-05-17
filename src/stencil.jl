"""
    LinearStencil{D,K,T,N,C<:NTuple{K,AbstractArray{T,N}}}

A variable-coefficient stencil aligned with mesh dimension `D`. For column `c`
at mesh position `p_c` and offset `Δ_k`, the matrix entry lands on the row at
mesh position `p_c − Δ_k` and carries coefficient `coefs[k][p_c]` — i.e., coefs
are anchored at the **column** mesh position (see `AGENTS.md`).

# Type parameters

- `D::Int` — mesh dimension on which the stencil acts (1-based). Must satisfy
  `1 ≤ D ≤ N`; enforced at construction.
- `K::Int` — number of stencil terms.
- `T` — shared element type of every coef array.
- `N` — coef-array dimensionality; matches the row/col
  `NTuple{N, AbstractUnitRange{Int}}` at assembly time.
- `C` — concrete tuple type of the coef containers; inferred from the
  constructor call. Heterogeneous containers (e.g. `Fill` + `Vector`) are fine
  as long as they share `eltype` and `ndims`.

# Fields

- `offsets::NTuple{K,Int}` — strictly descending 1-D offsets along dim `D`.
- `coefs::C` — `NTuple{K,<:AbstractArray{T,N}}` of coefficient arrays.

# Construction

    LinearStencil{D}(offsets, coefs)

Inner constructor (well-typed path) validates `D ≥ 1`, `D ≤ N`, and
strict-descending offsets via `issorted(offsets; lt = >=)`. The shared
`eltype`/`ndims` of `coefs` are enforced by the method signature.

A catch-all outer constructor (`LinearStencil{D}(::Tuple, ::Tuple)`) reports
friendly errors when the inputs are ill-typed (length mismatch, non-`Int`
offsets, non-`AbstractArray` coefs, mixed `eltype`, or mixed `ndims`).

# Examples

```julia
using FillArrays
n = 5

# Forward x-difference, constant coefs via Fill.
forward_x = LinearStencil{1}((1, 0), (Fill(1.0, n), Fill(-1.0, n)))
J = build(forward_x, (1:n,), (1:n,))    # one-shot assemble + update!

# Variable-coef: density-weighted gradient ψ[i] = (φ[i] − φ[i−1]) / ρ[i].
ρ = rand(n)
grad = LinearStencil{1}((0, -1), (1 ./ ρ, -1 ./ ρ))
```

See [`assemble`](@ref), [`update!`](@ref), [`build`](@ref) for matrix
construction.
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
        D <= N || throw(ArgumentError(
            "stencil dimension D=$D exceeds coef-array dimension N=$N"))
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
    _pattern!(rowval, colptr, offsets, row, col)

1-D pattern kernel. Fills `colptr` and `rowval` for the sparsity pattern
induced by `offsets` between the row and column index ranges.

# Algorithm

The per-column nnz count `q(c) = #{k : c_lo_k ≤ c ≤ c_hi_k}` (with
`c_lo_k = max(cmin, rmin + offsets[k])`, `c_hi_k = min(cmax, rmax + offsets[k])`)
is **piecewise constant** with at most `2K` breakpoints. Because offsets
are strictly descending, both endpoints are non-increasing in `k`, so
walking `k = k_last, …, k_first` yields two non-decreasing event streams
(lo at `c_lo_k`, hi at `c_hi_k + 1`) — no sort needed. Empty c-ranges
(offsets too large/small to reach any column) form a prefix and suffix
in `k` and are trimmed up front to `[k_first, k_last]`.

Within each constant-active segment with start column `prev`, start
pointer `cur`, slope `active = i_hi − i_lo`, and active offsets
`k_a:i_hi` (`k_a = i_lo + 1`), the writes are **closed-form pure
functions** of the loop indices — no read-modify-write of `colptr`, no
sequential dependency:

    colptr[c − cmin + 2]                                  = cur + active * (c − prev + 1)
    rowval[cur + active * (c − prev) + (k − k_a)]         = c − offsets[k] − rmin + 1

# CSC sortedness

For fixed `c`, slots are ascending in `k`, and the emitted row
`c − offsets[k]` is ascending in `k` (offsets strictly descending) —
the `SparseMatrixCSC` invariant, without a sort.

# Allocation

`Vector{Int}` `rowval` is `resize!`d once to the exact final length
(computed analytically from the trimmed offsets); no `push!`, no scratch
buffer.
"""
function _pattern!(
    rowval::Vector{Int},
    colptr::Vector{Int},
    offsets::NTuple{K,Int},
    row::AbstractUnitRange{Int},
    col::AbstractUnitRange{Int},
) where {K}
    rmin, rmax = first(row), last(row)
    cmin, cmax = first(col), last(col)
    colptr[1] = 1

    # Precompute event positions per offset.
    lo    = ntuple(k -> max(cmin, rmin + offsets[k]),     Val(K))
    hi_p1 = ntuple(k -> min(cmax, rmax + offsets[k]) + 1, Val(K))

    # Trim k to the non-empty range. Offsets strictly descending → empty
    # c-ranges form a prefix (offsets too large) and a suffix (too small).
    k_first = 1
    while k_first <= K && offsets[k_first] > cmax - rmin
        k_first += 1
    end
    k_last = K
    while k_last >= k_first && offsets[k_last] < cmin - rmax
        k_last -= 1
    end

    # Total nnz from precomputed bounds.
    total = 0
    for k in k_first:k_last
        total += hi_p1[k] - lo[k]
    end
    resize!(rowval, total)

    # Two-pointer segment walk over the precomputed event arrays. Sentinel
    # sits beyond any valid event column (≤ cmax + 1).
    sentinel = cmax + 2
    i_lo = k_last
    i_hi = k_last
    prev = cmin
    cur = 1
    while i_lo >= k_first || i_hi >= k_first
        pos_lo = i_lo >= k_first ? lo[i_lo]    : sentinel
        pos_hi = i_hi >= k_first ? hi_p1[i_hi] : sentinel
        e = min(pos_lo, pos_hi)
        active = i_hi - i_lo
        k_a = i_lo + 1
        for c in prev:(e - 1)
            colptr[c - cmin + 2] = cur + active * (c - prev + 1)
        end
        for c in prev:(e - 1), k in k_a:i_hi
            rowval[cur + active * (c - prev) + (k - k_a)] = c - offsets[k] - rmin + 1
        end
        cur += active * (e - prev)
        prev = e
        pos_lo == e && (i_lo -= 1)
        pos_hi == e && (i_hi -= 1)
    end
    # Tail (active = 0 once all events have fired): fill remaining colptr.
    for c in prev:cmax
        colptr[c - cmin + 2] = cur
    end
    return
end

"""
    _fill!(nzval, offsets, coefs, row, col)

1-D fill kernel. Same segment walk as [`_pattern!`](@ref): the slot
positions for `nzval` are a pure closed-form function of `(c, k)` within
each constant-active segment, so `colptr` is not consulted and the
matrix is consistent throughout the call.

For each constant-active segment with start column `prev`, start slot
`cur`, slope `active`, and active offsets `k_a:i_hi`:

    nzval[cur + active * (c − prev) + (k − k_a)] = coefs[k][c]

Allocation-free apart from whatever `getindex` on the user's coef arrays
costs (`Vector`, `Fill`, `OffsetArray` etc. are O(1)).
"""
function _fill!(
    nzval::AbstractVector{T},
    offsets::NTuple{K,Int},
    coefs::NTuple{K,AbstractArray{T,1}},
    row::AbstractUnitRange{Int},
    col::AbstractUnitRange{Int},
) where {K,T}
    rmin, rmax = first(row), last(row)
    cmin, cmax = first(col), last(col)

    lo    = ntuple(k -> max(cmin, rmin + offsets[k]),     Val(K))
    hi_p1 = ntuple(k -> min(cmax, rmax + offsets[k]) + 1, Val(K))

    k_first = 1
    while k_first <= K && offsets[k_first] > cmax - rmin
        k_first += 1
    end
    k_last = K
    while k_last >= k_first && offsets[k_last] < cmin - rmax
        k_last -= 1
    end

    sentinel = cmax + 2
    i_lo = k_last
    i_hi = k_last
    prev = cmin
    cur = 1
    while i_lo >= k_first || i_hi >= k_first
        pos_lo = i_lo >= k_first ? lo[i_lo]    : sentinel
        pos_hi = i_hi >= k_first ? hi_p1[i_hi] : sentinel
        e = min(pos_lo, pos_hi)
        active = i_hi - i_lo
        k_a = i_lo + 1
        for c in prev:(e - 1), k in k_a:i_hi
            nzval[cur + active * (c - prev) + (k - k_a)] = coefs[k][c]
        end
        cur += active * (e - prev)
        prev = e
        pos_lo == e && (i_lo -= 1)
        pos_hi == e && (i_hi -= 1)
    end
    return
end

"""
    assemble(st::LinearStencil{1,K,T,1},
             row::NTuple{1,AbstractUnitRange{Int}},
             col::NTuple{1,AbstractUnitRange{Int}}) -> SparseMatrixCSC{T,Int}

Build the sparsity pattern (`colptr`, `rowval`) of the operator induced by
`st` between the row and column index sets, and allocate `nzval`
**uninitialised**. Call `update!` to populate `nzval` before using the matrix
— or use [`build`](@ref) to do both in one shot.

`row` and `col` are interpreted on a *shared* integer mesh; for a column at
position `c ∈ col[1]` and offset `Δ_k`, the row position is `c − Δ_k`, and the
entry is emitted iff that position lies in `row[1]`.

Dispatch pins `D = 1` and `N = 1`; misuse turns into `MethodError`. The
`D ≤ N` invariant at the `LinearStencil` constructor catches the most common
misuse case (e.g. `LinearStencil{2}` with 1-D coefs) earlier with a friendly
`ArgumentError`.
"""
function assemble(
    st::LinearStencil{1,K,T,1},
    row::NTuple{1,AbstractUnitRange{Int}},
    col::NTuple{1,AbstractUnitRange{Int}},
) where {K,T}
    m, n = length(row[1]), length(col[1])
    colptr = Vector{Int}(undef, n + 1); colptr[1] = 1
    rowval = Int[]
    _pattern!(rowval, colptr, st.offsets, row[1], col[1])
    nzval = Vector{T}(undef, length(rowval))
    SparseMatrixCSC{T,Int}(m, n, colptr, rowval, nzval)
end

"""
    update!(mat::SparseMatrixCSC{T,Int}, st::LinearStencil{1,K,T,1},
            row::NTuple{1,AbstractUnitRange{Int}},
            col::NTuple{1,AbstractUnitRange{Int}}) -> mat

Write `mat.nzval` in place by re-walking `row`/`col` with `st`. `mat` must
have been produced by a matching `assemble(st, row, col)` call (same stencil,
same row/col) so its `colptr`/`rowval` align with the kernel's sweep.

Dispatch pins `D = 1` and `N = 1`; misuse turns into `MethodError`.
"""
function update!(
    mat::SparseMatrixCSC{T,Int},
    st::LinearStencil{1,K,T,1},
    row::NTuple{1,AbstractUnitRange{Int}},
    col::NTuple{1,AbstractUnitRange{Int}},
) where {T,K}
    _fill!(mat.nzval, st.offsets, st.coefs, row[1], col[1])
    return mat
end

"""
    build(st::LinearStencil, row, col) -> SparseMatrixCSC

Equivalent to `update!(assemble(st, row, col), st, row, col)`. Use when you
don't need the pattern/values split — the returned matrix is fully populated
and ready to use.
"""
build(st::LinearStencil, row, col) = update!(assemble(st, row, col), st, row, col)
