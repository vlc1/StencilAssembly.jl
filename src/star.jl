# `StarStencil` is defined in StencilCore (interlaced single-coefficient
# format: one `SVector{M}` per cell holding the whole star, `M = 2NL+1`, in
# reverse-lex offset order with the diagonal as the explicit middle slot).
# This file adds the CSC assembly methods, which dispatch on a concrete-array
# coefficient and `S = ColumnAccess`; the 1-D case delegates to `LinearStencil`
# via `_as_linear` (the 1-D layout coincides with `LinearStencil`'s).

"""
    _as_linear(st::StarStencil{L, 1, M, E, A, S}) -> LinearStencil{1, …, S}

Convert a 1-D `StarStencil` to the equivalent `LinearStencil{1}` with offsets
`−L … +L` and the same coefficient. The 1-D interlaced layout is already the
ascending-offset `SVector` a `LinearStencil` expects, so the coefficient is
reused verbatim. Preserves the access style.
"""
_as_linear(st::StarStencil{L, 1, M, E, A, S}) where {L, M, E, A, S} =
    LinearStencil{1}(S, SUnitRange(-L, L), st.term)

"""
    assemble(st::StarStencil{L, 1, M, SVector{M, T}, A, ColumnAccess}, row, col) -> SparseMatrixCSC{T, Int}

1-D `StarStencil` assembly delegates to the equivalent `LinearStencil{1}`.
"""
assemble(
    st::StarStencil{L, 1, M, SVector{M, T}, A, ColumnAccess},
    row::NTuple{1, AbstractUnitRange{Int}},
    col::NTuple{1, AbstractUnitRange{Int}},
) where {L, M, T, A<:AbstractArray{SVector{M, T}, 1}} =
    assemble(_as_linear(st), row, col)

"""
    update!(mat, st::StarStencil{L, 1, M, SVector{M, T}, A, ColumnAccess}, row, col) -> mat

1-D `StarStencil` update delegates to the equivalent `LinearStencil{1}`.
"""
update!(
    mat::SparseMatrixCSC{T, Int},
    st::StarStencil{L, 1, M, SVector{M, T}, A, ColumnAccess},
    row::NTuple{1, AbstractUnitRange{Int}},
    col::NTuple{1, AbstractUnitRange{Int}},
) where {L, T, M, A<:AbstractArray{SVector{M, T}, 1}} =
    update!(mat, _as_linear(st), row, col)

# Per-axis row-linear-index strides: s[d] = prod(length(row[e]) for e in 1:d-1).
@inline function _row_strides(row::NTuple{N, AbstractUnitRange{Int}}) where {N}
    ntuple(Val(N)) do d
        s = 1
        for e in 1:(d - 1)
            s *= length(row[e])
        end
        s
    end
end

# Per-axis guard: 2L ≤ length(row[d]) for every d. Ensures the reverse-lex
# offset order coincides with CSC-ascending rows (no sort) and that the
# three-phase boundary trimming is exact.
@inline function _star_guard(::Val{L}, row::NTuple{N, AbstractUnitRange{Int}}) where {L, N}
    for d in 1:N
        2L <= length(row[d]) || throw(ArgumentError(
            "stencil reach 2L=$(2L) exceeds length(row[$d])=$(length(row[d])); " *
            "the interlaced star kernel requires 2L ≤ length(row[d]) for the " *
            "reverse-lex offset order to match CSC row order"))
    end
    return nothing
end

"""
    assemble(st::StarStencil{L, N, M, SVector{M, T}, A, ColumnAccess}, row, col) -> SparseMatrixCSC{T, Int}

N-D entry (`N ≥ 2`, `S = ColumnAccess`). Enforces the per-axis guard
`2L ≤ length(row[d])`. Builds `colptr` / `rowval` and allocates uninitialised
`nzval`; call [`update!`](@ref) to populate, or use [`build`](@ref).
"""
function assemble(
    st::StarStencil{L, N, M, SVector{M, T}, A, ColumnAccess},
    row::NTuple{N, AbstractUnitRange{Int}},
    col::NTuple{N, AbstractUnitRange{Int}},
) where {L, N, M, T, A<:AbstractArray{SVector{M, T}, N}}
    _star_guard(Val(L), row)
    m = prod(length, row); n = prod(length, col)
    colptr = Vector{Int}(undef, n + 1)
    rowval = Int[]
    _pattern_nd_star!(rowval, colptr, Val(L), Val(N), row, col)
    nzval = Vector{T}(undef, length(rowval))
    SparseMatrixCSC{T, Int}(m, n, colptr, rowval, nzval)
end

"""
    update!(mat, st::StarStencil{L, N, M, SVector{M, T}, A, ColumnAccess}, row, col) -> mat

N-D in-place value update (`N ≥ 2`, `S = ColumnAccess`); same guard as
[`assemble`](@ref). `mat` must come from a matching `assemble`.
"""
function update!(
    mat::SparseMatrixCSC{T, Int},
    st::StarStencil{L, N, M, SVector{M, T}, A, ColumnAccess},
    row::NTuple{N, AbstractUnitRange{Int}},
    col::NTuple{N, AbstractUnitRange{Int}},
) where {L, T, N, M, A<:AbstractArray{SVector{M, T}, N}}
    _star_guard(Val(L), row)
    _fill_nd_star!(mat.nzval, Val(L), Val(N), st.term, row, col)
    return mat
end

"""
    _pattern_nd_star!(rowval, colptr, ::Val{L}, ::Val{N}, row, col)

N-D interlaced star pattern kernel. Iterates output columns in column-major
order; for each column walks the `M = 2NL+1` canonical offsets in CSC-ascending
row order (upper block `d = N..1, o = L..1`; diagonal; lower block
`d = 1..N, o = 1..L`), trimming off-mesh slots. The row linear index of a slot
is `base − o·s[d]`, where `base` is the diagonal row index and `s` the per-axis
strides; under the guard this is monotonic in slot order, so no sort is needed.

A column with one axis `g` outside `row[g]` emits only that axis's slots; with
two or more outside, nothing.
"""
function _pattern_nd_star!(
    rowval::Vector{Int},
    colptr::Vector{Int},
    ::Val{L}, ::Val{N},
    row::NTuple{N, AbstractUnitRange{Int}},
    col::NTuple{N, AbstractUnitRange{Int}},
) where {L, N}
    s = _row_strides(row)
    colptr[1] = 1
    cur = 1
    j = 0
    @inbounds for C in CartesianIndices(col)
        c = Tuple(C)
        base = 1
        n_invalid = 0
        g = 0
        for e in 1:N
            ce = c[e]
            base += (ce - first(row[e])) * s[e]
            if !(ce in row[e])
                n_invalid += 1
                g = e
            end
        end
        if n_invalid <= 1
            # Upper block (rows above the diagonal): ascending row.
            for d in N:-1:1
                if n_invalid == 0 || d == g
                    sd = s[d]; cd = c[d]
                    for o in L:-1:1
                        if (cd - o) in row[d]
                            push!(rowval, base - o * sd)
                            cur += 1
                        end
                    end
                end
            end
            # Diagonal.
            if n_invalid == 0
                push!(rowval, base)
                cur += 1
            end
            # Lower block (rows below the diagonal): shift −o.
            for d in 1:N
                if n_invalid == 0 || d == g
                    sd = s[d]; cd = c[d]
                    for o in 1:L
                        if (cd + o) in row[d]
                            push!(rowval, base + o * sd)
                            cur += 1
                        end
                    end
                end
            end
        end
        j += 1
        colptr[j + 1] = cur
    end
    return
end

"""
    _fill_nd_star!(nzval, ::Val{L}, ::Val{N}, term, row, col)

N-D interlaced star fill kernel. Same per-column slot walk as
[`_pattern_nd_star!`](@ref); fetches the column's coefficient `SVector` once
(`sv = term[C]`) and writes `sv[k]`, with `k` the slot's index in the
reverse-lex storage order (diagonal at `NL+1`). Allocation-free.
"""
function _fill_nd_star!(
    nzval::AbstractVector,
    ::Val{L}, ::Val{N},
    term::AbstractArray{<:SVector, N},
    row::NTuple{N, AbstractUnitRange{Int}},
    col::NTuple{N, AbstractUnitRange{Int}},
) where {L, N}
    NL = N * L
    idx = 1
    @inbounds for C in CartesianIndices(col)
        c = Tuple(C)
        n_invalid = 0
        g = 0
        for e in 1:N
            if !(c[e] in row[e])
                n_invalid += 1
                g = e
            end
        end
        if n_invalid <= 1
            sv = term[C]
            for d in N:-1:1
                if n_invalid == 0 || d == g
                    cd = c[d]
                    for o in L:-1:1
                        if (cd - o) in row[d]
                            nzval[idx] = sv[NL + 1 + (d - 1) * L + o]
                            idx += 1
                        end
                    end
                end
            end
            if n_invalid == 0
                nzval[idx] = sv[NL + 1]
                idx += 1
            end
            for d in 1:N
                if n_invalid == 0 || d == g
                    cd = c[d]
                    for o in 1:L
                        if (cd + o) in row[d]
                            nzval[idx] = sv[(N - d) * L + (L - o + 1)]
                            idx += 1
                        end
                    end
                end
            end
        end
    end
    return
end
