# `StarStencil` is defined in StencilCore (its coefficient containers may be
# concrete arrays or symbolic terms). This file adds the CSC assembly methods,
# which dispatch on concrete-array coefficients and `S = ColumnAccess`; the
# 1-D case delegates to `LinearStencil` via `_as_linear`.

"""
    _as_linear(st::StarStencil{L, 1, M, E, C, S}) -> LinearStencil{1, …, S}

Convert a 1-D `StarStencil` to the equivalent `LinearStencil{1}` with
offsets `−L … +L` and the same per-cell coefficient arrays. Preserves
the access style. The 1-D `assemble` / `update!` methods for
`StarStencil` delegate through this.
"""
_as_linear(st::StarStencil{L, 1, M, E, C, S}) where {L, M, E, C, S} =
    LinearStencil{1}(S, SUnitRange(-L, L), st.terms[1])

"""
    assemble(st::StarStencil{L, 1, M, SVector{M, T}, C, ColumnAccess}, row, col) -> SparseMatrixCSC{T, Int}

1-D `StarStencil` (with `ColumnAccess`) assembly delegates to the
equivalent `LinearStencil{1, …, ColumnAccess}` — no parallel codepath.
"""
assemble(
    st::StarStencil{L, 1, M, SVector{M, T}, C, ColumnAccess},
    row::NTuple{1, AbstractUnitRange{Int}},
    col::NTuple{1, AbstractUnitRange{Int}},
) where {L, M, T, C<:NTuple{1, AbstractArray{SVector{M, T}, 1}}} =
    assemble(_as_linear(st), row, col)

"""
    update!(mat, st::StarStencil{L, 1, M, SVector{M, T}, C, ColumnAccess}, row, col) -> mat

1-D `StarStencil` (with `ColumnAccess`) update delegates to the
equivalent `LinearStencil{1, …, ColumnAccess}`.
"""
update!(
    mat::SparseMatrixCSC{T, Int},
    st::StarStencil{L, 1, M, SVector{M, T}, C, ColumnAccess},
    row::NTuple{1, AbstractUnitRange{Int}},
    col::NTuple{1, AbstractUnitRange{Int}},
) where {L, T, M, C<:NTuple{1, AbstractArray{SVector{M, T}, 1}}} =
    update!(mat, _as_linear(st), row, col)

"""
    assemble(st::StarStencil{L, N, M, SVector{M, T}, C, ColumnAccess}, row, col) -> SparseMatrixCSC{T, Int}

N-D entry (`N ≥ 2`, `S = ColumnAccess`). Enforces the per-axis guard
`2L ≤ length(row[d])` for every `d ∈ 1:N` — this is the LinearStencil
correctness bound applied independently per axis, and it also implies
the cross-axis row-ordering invariant the kernel relies on (axis-d's
row block lives entirely outside axis-(d+1)'s, so per-column CSC
sortedness comes from concatenation alone, no merge needed).

Builds `colptr` / `rowval` and allocates uninitialised `nzval`; call
[`update!`](@ref) to populate values, or use [`build`](@ref) to do both.
"""
function assemble(
    st::StarStencil{L, N, M, SVector{M, T}, C, ColumnAccess},
    row::NTuple{N, AbstractUnitRange{Int}},
    col::NTuple{N, AbstractUnitRange{Int}},
) where {L, N, M, T, C<:NTuple{N, AbstractArray{SVector{M, T}, N}}}
    for d in 1:N
        2L <= length(row[d]) || throw(ArgumentError(
            "stencil reach 2L=$(2L) exceeds length(row[$d])=$(length(row[d])); " *
            "this would force the saturated-middle phase in the per-axis " *
            "three-phase kernel, which is out of scope"))
    end
    m = prod(length, row); n = prod(length, col)
    colptr = Vector{Int}(undef, n + 1); colptr[1] = 1
    rowval = Int[]
    _pattern_nd_star!(rowval, colptr, Val(L), row, col, Val(N),
                      1, 1, 0, 0, 0, ())
    nzval = Vector{T}(undef, length(rowval))
    SparseMatrixCSC{T, Int}(m, n, colptr, rowval, nzval)
end

"""
    update!(mat, st::StarStencil{L, N, M, SVector{M, T}, C, ColumnAccess}, row, col) -> mat

N-D in-place value update (`N ≥ 2`, `S = ColumnAccess`); carries the
same per-axis guard as [`assemble`](@ref). `mat` must have been
produced by a matching `assemble` so its `colptr`/`rowval` align with
the kernel's sweep.
"""
function update!(
    mat::SparseMatrixCSC{T, Int},
    st::StarStencil{L, N, M, SVector{M, T}, C, ColumnAccess},
    row::NTuple{N, AbstractUnitRange{Int}},
    col::NTuple{N, AbstractUnitRange{Int}},
) where {L, T, N, M, C<:NTuple{N, AbstractArray{SVector{M, T}, N}}}
    for d in 1:N
        2L <= length(row[d]) || throw(ArgumentError(
            "stencil reach 2L=$(2L) exceeds length(row[$d])=$(length(row[d])); " *
            "this would force the saturated-middle phase in the per-axis " *
            "three-phase kernel, which is out of scope"))
    end
    _fill_nd_star!(mat.nzval, Val(L), st.terms, row, col, Val(N),
                   1, 0, 0, (), ())
    return mat
end

"""
    _pattern_nd_star!(rowval, colptr, ::Val{L}, row, col, ::Val{Nd},
                      cur, col_j, row_base, n_outer_invalid,
                      invalid_outer_d, axis_state) -> (cur, col_j)

Recursive N-D star pattern kernel. Peels dimensions outermost (last) →
innermost (first); each call returns updated `(cur, col_j)` state so
output columns are visited exactly once.

Threaded state:
- `cur`, `col_j`: next free index in `rowval` and CSC columns finalised.
- `row_base`: row-linear-index shift from peeled dims,
  `Σ_{d peeled} (c_d − rmin_d) · s_d` — accumulated for **every** peeled
  dim regardless of validity (negative or out-of-range values are
  corrected by the kernel subtracting `δ · s_d` at the base).
- `n_outer_invalid`: count of peeled dims with `c_d ∉ row[d]`
  (0, 1, or `≥ 2` short-circuit).
- `invalid_outer_d`: the single invalid peeled dim's index (meaningful
  only when `n_outer_invalid == 1`).
- `axis_state::NTuple{N_outer, NTuple{3, Int}}`: per peeled dim,
  `(s_d, δ_lo_d_c, δ_hi_d_c)` — stride and trimmed offset range.
  Prepended on each peel, so `axis_state[i]` corresponds to dim `i + 1`.

Per-column branching at the base (`Val{1}`):

| `n_outer_invalid` | `c_1 ∈ row[1]` | Emission                       |
|-------------------|----------------|--------------------------------|
| 0                 | true           | Full star (all axes + center)  |
| 0                 | false          | Axis-1 only                    |
| 1                 | true           | Axis-`invalid_outer_d` only    |
| 1                 | false          | Nothing                        |
| ≥ 2               | —              | Nothing                        |

Under the per-axis guard, the full-star block emission is row-ascending
by construction: axes from N down to 2 above-center, axis 1 above,
center, axis 1 below, axes 2 up to N below.
"""
function _pattern_nd_star! end

@inline function _pattern_nd_star!(
    rowval::Vector{Int},
    colptr::Vector{Int},
    ::Val{L},
    row::NTuple{1, AbstractUnitRange{Int}},
    col::NTuple{1, AbstractUnitRange{Int}},
    ::Val{1},
    cur::Int, col_j::Int,
    row_base::Int,
    n_outer_invalid::Int,
    invalid_outer_d::Int,
    axis_state::NTuple{N_outer, NTuple{3, Int}},
)::Tuple{Int, Int} where {L, N_outer}
    rmin, rmax = first(row[1]), last(row[1])
    cmin, cmax = first(col[1]), last(col[1])

    for c_1 in cmin:cmax
        c_1_valid = c_1 >= rmin && c_1 <= rmax
        δ_hi_1 = min(L, c_1 - rmin)
        δ_lo_1 = max(-L, c_1 - rmax)
        r_anchor = row_base + (c_1 - rmin) + 1

        if n_outer_invalid == 0 && c_1_valid
            # Full star.
            for i in N_outer:-1:1
                s_d, δ_lo_d, δ_hi_d = axis_state[i]
                for δ in δ_hi_d:-1:max(1, δ_lo_d)
                    push!(rowval, r_anchor - δ * s_d)
                    cur += 1
                end
            end
            for δ in δ_hi_1:-1:max(1, δ_lo_1)
                push!(rowval, r_anchor - δ)
                cur += 1
            end
            push!(rowval, r_anchor)
            cur += 1
            for δ in min(-1, δ_hi_1):-1:δ_lo_1
                push!(rowval, r_anchor - δ)
                cur += 1
            end
            for i in 1:N_outer
                s_d, δ_lo_d, δ_hi_d = axis_state[i]
                for δ in min(-1, δ_hi_d):-1:δ_lo_d
                    push!(rowval, r_anchor - δ * s_d)
                    cur += 1
                end
            end
        elseif n_outer_invalid == 0 && !c_1_valid
            # Axis-1 only (c_1 ∉ row[1] ⇒ δ = 0 is automatically out of [δ_lo_1, δ_hi_1]).
            for δ in δ_hi_1:-1:δ_lo_1
                push!(rowval, r_anchor - δ)
                cur += 1
            end
        elseif n_outer_invalid == 1 && c_1_valid
            # Axis-invalid_outer_d only.
            s_d, δ_lo_d, δ_hi_d = axis_state[invalid_outer_d - 1]
            for δ in δ_hi_d:-1:δ_lo_d
                push!(rowval, r_anchor - δ * s_d)
                cur += 1
            end
        end
        # else: nothing emitted.

        colptr[col_j + 1] = cur
        col_j += 1
    end
    return (cur, col_j)
end

@inline function _pattern_nd_star!(
    rowval::Vector{Int},
    colptr::Vector{Int},
    ::Val{L},
    row::NTuple{Nd, AbstractUnitRange{Int}},
    col::NTuple{Nd, AbstractUnitRange{Int}},
    ::Val{Nd},
    cur::Int, col_j::Int,
    row_base::Int,
    n_outer_invalid::Int,
    invalid_outer_d::Int,
    axis_state::NTuple{N_outer, NTuple{3, Int}},
)::Tuple{Int, Int} where {L, Nd, N_outer}
    row_last = last(row); col_last = last(col)
    row_rest = Base.front(row); col_rest = Base.front(col)
    rmin_last, rmax_last = first(row_last), last(row_last)
    cmin_last, cmax_last = first(col_last), last(col_last)

    s_Nd = prod(length, row_rest; init=1)
    cols_per_iter = prod(length, col_rest; init=1)

    for c_Nd in cmin_last:cmax_last
        c_Nd_valid = c_Nd >= rmin_last && c_Nd <= rmax_last
        δ_hi_Nd = min(L, c_Nd - rmin_last)
        δ_lo_Nd = max(-L, c_Nd - rmax_last)

        new_row_base = row_base + (c_Nd - rmin_last) * s_Nd
        new_axis_state = ((s_Nd, δ_lo_Nd, δ_hi_Nd), axis_state...)

        if c_Nd_valid
            cur, col_j = _pattern_nd_star!(
                rowval, colptr, Val(L), row_rest, col_rest, Val(Nd - 1),
                cur, col_j, new_row_base,
                n_outer_invalid, invalid_outer_d, new_axis_state,
            )
        else
            new_n_outer_invalid = n_outer_invalid + 1
            if new_n_outer_invalid >= 2
                # No emissions possible from any subcolumn under this c_Nd.
                for _ in 1:cols_per_iter
                    colptr[col_j + 1] = cur
                    col_j += 1
                end
            else
                cur, col_j = _pattern_nd_star!(
                    rowval, colptr, Val(L), row_rest, col_rest, Val(Nd - 1),
                    cur, col_j, new_row_base,
                    new_n_outer_invalid, Nd, new_axis_state,
                )
            end
        end
    end
    return (cur, col_j)
end

"""
    _fill_nd_star!(nzval, ::Val{L}, terms, row, col, ::Val{Nd},
                   nzval_idx, n_outer_invalid, invalid_outer_d,
                   axis_state, outer_coords) -> nzval_idx

Recursive N-D star fill kernel. Mirrors [`_pattern_nd_star!`](@ref) but
writes only `nzval` (`colptr`/`rowval` were finalised by `assemble`).
Threads `outer_coords` (mesh positions of peeled dims, ordered to match
coef array axes). At the base, the full-star branch fetches each axis's
per-column `SVector` once (`svs = ntuple(d -> terms[d][c_1,
outer_coords...], N)`) and reads slot `k` from it; the center slot writes
`Σ_d svs[d][L + 1]` (diagonal merge). The single-axis branches fetch only
the one axis vector they emit from.
"""
function _fill_nd_star! end

@inline function _fill_nd_star!(
    nzval::AbstractVector{T},
    ::Val{L},
    terms::NTuple{N, AbstractArray{SVector{M, T}, N}},
    row::NTuple{1, AbstractUnitRange{Int}},
    col::NTuple{1, AbstractUnitRange{Int}},
    ::Val{1},
    nzval_idx::Int,
    n_outer_invalid::Int,
    invalid_outer_d::Int,
    axis_state::NTuple{N_outer, NTuple{3, Int}},
    outer_coords::NTuple{N_outer, Int},
)::Int where {L, T, N, M, N_outer}
    rmin, rmax = first(row[1]), last(row[1])
    cmin, cmax = first(col[1]), last(col[1])

    for c_1 in cmin:cmax
        c_1_valid = c_1 >= rmin && c_1 <= rmax
        δ_hi_1 = min(L, c_1 - rmin)
        δ_lo_1 = max(-L, c_1 - rmax)

        if n_outer_invalid == 0 && c_1_valid
            # Full star — fetch each axis's coefficient vector once.
            svs = ntuple(d -> terms[d][c_1, outer_coords...], Val(N))
            for i in N_outer:-1:1
                d = i + 1
                _, δ_lo_d, δ_hi_d = axis_state[i]
                for δ in δ_hi_d:-1:max(1, δ_lo_d)
                    nzval[nzval_idx] = svs[d][δ + L + 1]
                    nzval_idx += 1
                end
            end
            for δ in δ_hi_1:-1:max(1, δ_lo_1)
                nzval[nzval_idx] = svs[1][δ + L + 1]
                nzval_idx += 1
            end
            center_val = zero(T)
            for d in 1:N
                center_val += svs[d][L + 1]
            end
            nzval[nzval_idx] = center_val
            nzval_idx += 1
            for δ in min(-1, δ_hi_1):-1:δ_lo_1
                nzval[nzval_idx] = svs[1][δ + L + 1]
                nzval_idx += 1
            end
            for i in 1:N_outer
                d = i + 1
                _, δ_lo_d, δ_hi_d = axis_state[i]
                for δ in min(-1, δ_hi_d):-1:δ_lo_d
                    nzval[nzval_idx] = svs[d][δ + L + 1]
                    nzval_idx += 1
                end
            end
        elseif n_outer_invalid == 0 && !c_1_valid
            sv1 = terms[1][c_1, outer_coords...]
            for δ in δ_hi_1:-1:δ_lo_1
                nzval[nzval_idx] = sv1[δ + L + 1]
                nzval_idx += 1
            end
        elseif n_outer_invalid == 1 && c_1_valid
            _, δ_lo_d, δ_hi_d = axis_state[invalid_outer_d - 1]
            svd = terms[invalid_outer_d][c_1, outer_coords...]
            for δ in δ_hi_d:-1:δ_lo_d
                nzval[nzval_idx] = svd[δ + L + 1]
                nzval_idx += 1
            end
        end
    end
    return nzval_idx
end

@inline function _fill_nd_star!(
    nzval::AbstractVector{T},
    ::Val{L},
    terms::NTuple{N, AbstractArray{SVector{M, T}, N}},
    row::NTuple{Nd, AbstractUnitRange{Int}},
    col::NTuple{Nd, AbstractUnitRange{Int}},
    ::Val{Nd},
    nzval_idx::Int,
    n_outer_invalid::Int,
    invalid_outer_d::Int,
    axis_state::NTuple{N_outer, NTuple{3, Int}},
    outer_coords::NTuple{N_outer, Int},
)::Int where {L, T, N, M, Nd, N_outer}
    row_last = last(row); col_last = last(col)
    row_rest = Base.front(row); col_rest = Base.front(col)
    rmin_last, rmax_last = first(row_last), last(row_last)
    cmin_last, cmax_last = first(col_last), last(col_last)

    s_Nd = prod(length, row_rest; init=1)

    for c_Nd in cmin_last:cmax_last
        c_Nd_valid = c_Nd >= rmin_last && c_Nd <= rmax_last
        δ_hi_Nd = min(L, c_Nd - rmin_last)
        δ_lo_Nd = max(-L, c_Nd - rmax_last)

        new_axis_state = ((s_Nd, δ_lo_Nd, δ_hi_Nd), axis_state...)
        new_outer_coords = (c_Nd, outer_coords...)

        if c_Nd_valid
            nzval_idx = _fill_nd_star!(
                nzval, Val(L), terms, row_rest, col_rest, Val(Nd - 1),
                nzval_idx, n_outer_invalid, invalid_outer_d,
                new_axis_state, new_outer_coords,
            )
        else
            new_n_outer_invalid = n_outer_invalid + 1
            if new_n_outer_invalid < 2
                nzval_idx = _fill_nd_star!(
                    nzval, Val(L), terms, row_rest, col_rest, Val(Nd - 1),
                    nzval_idx, new_n_outer_invalid, Nd,
                    new_axis_state, new_outer_coords,
                )
            end
            # n_outer_invalid >= 2: no emissions, no fill work.
        end
    end
    return nzval_idx
end
