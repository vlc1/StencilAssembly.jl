# `LinearStencil` is defined in StencilCore (its coefficient field may be a
# concrete array or a symbolic term). This file adds the CSC assembly methods,
# which dispatch on a *concrete-array* coefficient and `S = ColumnAccess`.

"""
    _pattern!(rowval, colptr, offsets::SUnitRange{O, L}, row, col)

1-D pattern kernel — three-phase contiguous walk. Fills `colptr` /
`rowval` for the sparsity pattern of the operator induced by `offsets`
between `row` and `col`.

Effective offset range after trim:

    δ_lo = max(O,         cmin − rmax)
    δ_hi = min(O + L − 1, cmax − rmin)

If `δ_lo > δ_hi` the operator is empty (early return). Otherwise:

| phase       | columns            | `active`                          | rows                             |
|-------------|--------------------|-----------------------------------|----------------------------------|
| Left ramp   | `[cmin, c_LR − 1]` | `max(0, c − rmin − δ_lo + 1)`     | `1, 2, …, active(c)`             |
| Interior    | `[c_LR, c_RR]`     | `Leff = δ_hi − δ_lo + 1` const.   | `r0 … r0 + Leff − 1`, stride 1   |
| Right ramp  | `[c_RR + 1, cmax]` | `max(0, δ_hi − c + rmax + 1)`     | `r0 … r0 + active − 1`, stride 1 |

with `c_LR = max(cmin, rmin + δ_hi)`, `c_RR = min(cmax, rmax + δ_lo)`,
`r0 = c − δ_hi − rmin + 1`. Each phase walks `δ` high → low so rows
ascend per column (CSC sortedness without sort). Interior uses
closed-form `cur(c) = cur_int_0 + Leff * (c − c_LR)` — column writes
independent. Ramps use `max(0, active)` to absorb off-mesh column tails.

Under the `L − 1 ≤ length(row)` guard (enforced by [`assemble`](@ref) /
[`update!`](@ref)), `c_LR ≤ c_RR + 1`: interior non-empty for
`L ≤ length(row)`, empty (ramps tile without gap) at exactly
`L = length(row) + 1`. `rowval` is `resize!`d once to the analytic nnz
(per-offset overlap summed); otherwise allocation-free.
"""
function _pattern!(
    rowval::Vector{Int},
    colptr::Vector{Int},
    offsets::SUnitRange{O, L},
    row::AbstractUnitRange{Int},
    col::AbstractUnitRange{Int},
) where {O, L}
    rmin, rmax = first(row), last(row)
    cmin, cmax = first(col), last(col)
    colptr[1] = 1

    δ_lo = max(O,         cmin - rmax)
    δ_hi = min(O + L - 1, cmax - rmin)

    if δ_lo > δ_hi
        resize!(rowval, 0)
        for c in cmin:cmax
            colptr[c - cmin + 2] = 1
        end
        return
    end

    Leff = δ_hi - δ_lo + 1
    c_LR = max(cmin, rmin + δ_hi)
    c_RR = min(cmax, rmax + δ_lo)

    # Total nnz: sum over effective offsets of overlap with col range.
    total = 0
    for δ in δ_lo:δ_hi
        total += min(cmax, rmax + δ) - max(cmin, rmin + δ) + 1
    end
    resize!(rowval, total)

    cur = 1

    # Left ramp.
    for c in cmin:(c_LR - 1)
        active = max(0, c - rmin - δ_lo + 1)
        for i in 0:(active - 1)
            rowval[cur + i] = i + 1
        end
        cur += active
        colptr[c - cmin + 2] = cur
    end

    # Interior: closed-form cur(c). Each c is independent.
    cur_int_0 = cur
    for c in c_LR:c_RR
        cur_c = cur_int_0 + Leff * (c - c_LR)
        r0    = c - δ_hi - rmin + 1
        for i in 0:(Leff - 1)
            rowval[cur_c + i] = r0 + i
        end
        colptr[c - cmin + 2] = cur_c + Leff
    end
    cur = cur_int_0 + Leff * max(0, c_RR - c_LR + 1)

    # Right ramp.
    for c in (c_RR + 1):cmax
        r0     = c - δ_hi - rmin + 1
        active = max(0, rmax - rmin + 1 - r0 + 1)
        for i in 0:(active - 1)
            rowval[cur + i] = r0 + i
        end
        cur += active
        colptr[c - cmin + 2] = cur
    end
    return
end

"""
    _fill!(nzval, offsets::SUnitRange{O, L}, term, row, col)

1-D fill kernel — same three-phase shape as [`_pattern!`](@ref); writes
only `nzval`, never touches `colptr`. Fetches the column's coefficient
`SVector` once per active column (`sv = term[c]`) and reads slot
`k = δ − O + 1` (ascending offset order). Allocation-free apart from the
single `term[c]` `getindex` per active column (O(1) for `Vector` / `Fill`
/ `OffsetArray`; an `SVector{L}` is isbits and returned by value).
"""
function _fill!(
    nzval::AbstractVector{T},
    offsets::SUnitRange{O, L},
    term::AbstractArray{SVector{L, T}, 1},
    row::AbstractUnitRange{Int},
    col::AbstractUnitRange{Int},
) where {O, L, T}
    rmin, rmax = first(row), last(row)
    cmin, cmax = first(col), last(col)

    δ_lo = max(O,         cmin - rmax)
    δ_hi = min(O + L - 1, cmax - rmin)
    δ_lo > δ_hi && return

    Leff = δ_hi - δ_lo + 1
    c_LR = max(cmin, rmin + δ_hi)
    c_RR = min(cmax, rmax + δ_lo)
    cur = 1

    # Left ramp: δ descends from c − rmin to δ_lo. k = δ − O + 1.
    for c in cmin:(c_LR - 1)
        active = max(0, c - rmin - δ_lo + 1)
        if active > 0
            sv = term[c]
            for i in 0:(active - 1)
                δ = (c - rmin) - i
                k = δ - O + 1
                nzval[cur + i] = sv[k]
            end
            cur += active
        end
    end

    # Interior: closed-form cur(c); δ descends from δ_hi to δ_lo.
    cur_int_0 = cur
    for c in c_LR:c_RR
        cur_c = cur_int_0 + Leff * (c - c_LR)
        sv = term[c]
        for i in 0:(Leff - 1)
            δ = δ_hi - i
            k = δ - O + 1
            nzval[cur_c + i] = sv[k]
        end
    end
    cur = cur_int_0 + Leff * max(0, c_RR - c_LR + 1)

    # Right ramp: δ descends from δ_hi to c − rmax.
    for c in (c_RR + 1):cmax
        r0     = c - δ_hi - rmin + 1
        active = max(0, rmax - rmin + 1 - r0 + 1)
        if active > 0
            sv = term[c]
            for i in 0:(active - 1)
                δ = δ_hi - i
                k = δ - O + 1
                nzval[cur + i] = sv[k]
            end
            cur += active
        end
    end
    return
end

"""
    assemble(st::LinearStencil{1, O, L, SVector{L, T}, A, ColumnAccess},
             row::NTuple{1, AbstractUnitRange{Int}},
             col::NTuple{1, AbstractUnitRange{Int}}) -> SparseMatrixCSC{T, Int}

Build the sparsity pattern (`colptr`, `rowval`) of the operator induced by
`st` between `row` and `col` and allocate `nzval` **uninitialised**. Call
[`update!`](@ref) to populate `nzval`, or use [`build`](@ref) to do both
in one shot.

`row` and `col` are interpreted on a shared integer mesh; the entry at
column `c ∈ col[1]` and offset `δ` lands on row `c − δ` iff it lies in
`row[1]`.

Dispatch pins `D = 1`, `N = 1`, and `S = ColumnAccess` (misuse →
`MethodError`); the `L − 1 ≤ length(row[1])` guard is the three-phase
kernel's exact correctness boundary.
"""
function assemble(
    st::LinearStencil{1, O, L, SVector{L, T}, A, ColumnAccess},
    row::NTuple{1, AbstractUnitRange{Int}},
    col::NTuple{1, AbstractUnitRange{Int}},
) where {O, L, T, A<:AbstractArray{SVector{L, T}, 1}}
    L - 1 <= length(row[1]) || throw(ArgumentError(
        "stencil width L=$L exceeds length(row[1])+1=$(length(row[1])+1); " *
        "the three-phase kernel is exact up to L = length(row) + 1, beyond which " *
        "the saturated-middle phase would be required and is out of scope"))
    m, n = length(row[1]), length(col[1])
    colptr = Vector{Int}(undef, n + 1); colptr[1] = 1
    rowval = Int[]
    _pattern!(rowval, colptr, st.offsets, row[1], col[1])
    nzval = Vector{T}(undef, length(rowval))
    SparseMatrixCSC{T, Int}(m, n, colptr, rowval, nzval)
end

"""
    update!(mat::SparseMatrixCSC{T, Int},
            st::LinearStencil{1, O, L, SVector{L, T}, A, ColumnAccess},
            row::NTuple{1, AbstractUnitRange{Int}},
            col::NTuple{1, AbstractUnitRange{Int}}) -> mat

Write `mat.nzval` in place by re-walking `row`/`col` with `st`. `mat` must
have been produced by a matching [`assemble`](@ref) (same `st`, `row`,
`col`) so its `colptr`/`rowval` align with the kernel's sweep. Carries the
same `L − 1 ≤ length(row[1])` guard.
"""
function update!(
    mat::SparseMatrixCSC{T, Int},
    st::LinearStencil{1, O, L, SVector{L, T}, A, ColumnAccess},
    row::NTuple{1, AbstractUnitRange{Int}},
    col::NTuple{1, AbstractUnitRange{Int}},
) where {T, O, L, A<:AbstractArray{SVector{L, T}, 1}}
    L - 1 <= length(row[1]) || throw(ArgumentError(
        "stencil width L=$L exceeds length(row[1])+1=$(length(row[1])+1); " *
        "the three-phase kernel is exact up to L = length(row) + 1, beyond which " *
        "the saturated-middle phase would be required and is out of scope"))
    _fill!(mat.nzval, st.offsets, st.term, row[1], col[1])
    return mat
end

# N-D assembly via state-threading recursive dimensional peeling.
# Each `_pattern_nd!` / `_fill_nd!` call returns updated state so each output
# column is visited exactly once; CSC sortedness falls out of the three-phase
# walk applied at the stencil dimension.

# Stride of dim D in the row linear index: s_D = prod(length(row[d]) for d in 1:D-1).
@inline function _row_stride(row::NTuple{N, AbstractUnitRange{Int}}, ::Val{D}) where {N, D}
    s = 1
    for d in 1:(D - 1)
        s *= length(row[d])
    end
    return s
end

function assemble(
    st::LinearStencil{D, O, L, SVector{L, T}, A, ColumnAccess},
    row::NTuple{N, AbstractUnitRange{Int}},
    col::NTuple{N, AbstractUnitRange{Int}},
) where {D, O, L, T, N, A<:AbstractArray{SVector{L, T}, N}}
    L - 1 <= length(row[D]) || throw(ArgumentError(
        "stencil width L=$L exceeds length(row[$D])+1=$(length(row[D])+1); " *
        "the three-phase kernel is exact up to L = length(row) + 1, beyond which " *
        "the saturated-middle phase would be required and is out of scope"))
    m, n = prod(length, row), prod(length, col)
    colptr = Vector{Int}(undef, n + 1); colptr[1] = 1
    rowval = Int[]
    s_D = _row_stride(row, Val(D))
    _pattern_nd!(rowval, colptr, st.offsets, row, col, Val(D), Val(N),
                 1, 1, 0, 1, 0, s_D)
    nzval = Vector{T}(undef, length(rowval))
    SparseMatrixCSC{T, Int}(m, n, colptr, rowval, nzval)
end

function update!(
    mat::SparseMatrixCSC{T, Int},
    st::LinearStencil{D, O, L, SVector{L, T}, A, ColumnAccess},
    row::NTuple{N, AbstractUnitRange{Int}},
    col::NTuple{N, AbstractUnitRange{Int}},
) where {D, O, L, T, N, A<:AbstractArray{SVector{L, T}, N}}
    L - 1 <= length(row[D]) || throw(ArgumentError(
        "stencil width L=$L exceeds length(row[$D])+1=$(length(row[D])+1); " *
        "the three-phase kernel is exact up to L = length(row) + 1, beyond which " *
        "the saturated-middle phase would be required and is out of scope"))
    _fill_nd!(mat.nzval, st.offsets, st.term, row, col, Val(D), Val(N),
              1, 1, 0, ())
    return mat
end

"""
    _pattern_nd!(rowval, colptr, offsets, row, col, ::Val{D}, ::Val{Nd},
                 cur, col_j, row_base, active, r_start, s_D) -> (cur, col_j)

Recursive N-D pattern kernel. Peels dimensions outermost (last) → innermost
(first); each call returns updated `(cur, col_j)` state so output columns
are visited exactly once.

Threaded state:
- `cur`: next free index in `rowval` (`length(rowval) + 1`).
- `col_j`: number of CSC columns already finalised (next column is `col_j`).
- `row_base`: accumulated row contribution from already-peeled non-D dims.
- `active`, `r_start`: number of valid offsets and dim-D row-index
  contribution at the smallest active row — set when peeling the stencil
  dim (`Nd == D`), consumed at the inner intersection base (`Nd == 1`,
  `D > 1`). Ignored at `(Nd, D) == (1, 1)`.
- `s_D`: stride of dim D in the row linear index (constant for the call).

Method dispatch on `(Val{D}, Val{Nd})`:
- `(Val{1}, Val{1})`: stencil base — three-phase walk on `row[1]` vs `col[1]`.
- `(Val{D}, Val{1})` with `D > 1`: inner intersection base — emit
  `active` rows per valid `c_1` as arithmetic sequence (step `s_D`).
- `(Val{Nd}, Val{Nd})` with `Nd ≥ 2`: stencil dim peel — three-phase trim
  on `col_Nd` vs `row_Nd`, recurse with per-`c_Nd` `(active, r_start)`.
- `(Val{D}, Val{Nd})` with `Nd ≥ 2`, `Nd ≠ D`: non-D dim peel — for each
  valid `c_Nd`, accumulate `row_base += (c_Nd − rmin_Nd)·s_Nd` and recurse.
"""
function _pattern_nd! end

# Method A: (Nd, D) = (1, 1) — stencil base case. Three-phase walk on the
# single dim with row_base added to each emitted row. active / r_start / s_D
# are placeholders (stencil is local).
@inline function _pattern_nd!(
    rowval::Vector{Int},
    colptr::Vector{Int},
    offsets::SUnitRange{O, L},
    row::NTuple{1, AbstractUnitRange{Int}},
    col::NTuple{1, AbstractUnitRange{Int}},
    ::Val{1}, ::Val{1},
    cur::Int, col_j::Int,
    row_base::Int, ::Int, ::Int, ::Int,
)::Tuple{Int, Int} where {O, L}
    rmin, rmax = first(row[1]), last(row[1])
    cmin, cmax = first(col[1]), last(col[1])

    δ_lo = max(O,         cmin - rmax)
    δ_hi = min(O + L - 1, cmax - rmin)

    if δ_lo > δ_hi
        for _ in cmin:cmax
            colptr[col_j + 1] = cur
            col_j += 1
        end
        return (cur, col_j)
    end

    Leff = δ_hi - δ_lo + 1
    c_LR = max(cmin, rmin + δ_hi)
    c_RR = min(cmax, rmax + δ_lo)

    # Left ramp.
    for c in cmin:(c_LR - 1)
        active_c = max(0, c - rmin - δ_lo + 1)
        for i in 0:(active_c - 1)
            push!(rowval, row_base + i + 1)
        end
        cur += active_c
        colptr[col_j + 1] = cur
        col_j += 1
    end

    # Interior.
    for c in c_LR:c_RR
        r0 = c - δ_hi - rmin + 1
        for i in 0:(Leff - 1)
            push!(rowval, row_base + r0 + i)
        end
        cur += Leff
        colptr[col_j + 1] = cur
        col_j += 1
    end

    # Right ramp.
    for c in (c_RR + 1):cmax
        r0 = c - δ_hi - rmin + 1
        active_c = max(0, rmax - rmin + 1 - r0 + 1)
        for i in 0:(active_c - 1)
            push!(rowval, row_base + r0 + i)
        end
        cur += active_c
        colptr[col_j + 1] = cur
        col_j += 1
    end

    return (cur, col_j)
end

# Method B: (Nd, D) = (1, D>1) — inner intersection base case. The stencil
# was applied at an outer peel, leaving `(active, r_start)` to thread here.
# Emit `active` rows per valid c_1 as arithmetic sequence with step s_D.
@inline function _pattern_nd!(
    rowval::Vector{Int},
    colptr::Vector{Int},
    offsets::SUnitRange{O, L},
    row::NTuple{1, AbstractUnitRange{Int}},
    col::NTuple{1, AbstractUnitRange{Int}},
    ::Val{D}, ::Val{1},
    cur::Int, col_j::Int,
    row_base::Int, active::Int, r_start::Int, s_D::Int,
)::Tuple{Int, Int} where {O, L, D}
    rmin, rmax = first(row[1]), last(row[1])
    cmin, cmax = first(col[1]), last(col[1])

    for c in cmin:cmax
        if c >= rmin && c <= rmax && active > 0
            r0 = row_base + (c - rmin) + r_start + 1
            for i in 0:(active - 1)
                push!(rowval, r0 + i * s_D)
            end
            cur += active
        end
        colptr[col_j + 1] = cur
        col_j += 1
    end

    return (cur, col_j)
end

# Method C: Nd ≥ 2, Nd == D — stencil dim peel. Apply three-phase trim to
# col_Nd vs row_Nd; for each c_Nd compute (active_c, r_start_c) and recurse.
@inline function _pattern_nd!(
    rowval::Vector{Int},
    colptr::Vector{Int},
    offsets::SUnitRange{O, L},
    row::NTuple{Nd, AbstractUnitRange{Int}},
    col::NTuple{Nd, AbstractUnitRange{Int}},
    ::Val{Nd}, ::Val{Nd},
    cur::Int, col_j::Int,
    row_base::Int, ::Int, ::Int, s_D::Int,
)::Tuple{Int, Int} where {O, L, Nd}
    row_last = last(row); col_last = last(col)
    row_rest = Base.front(row); col_rest = Base.front(col)
    rmin_last, rmax_last = first(row_last), last(row_last)
    cmin_last, cmax_last = first(col_last), last(col_last)

    δ_lo = max(O,         cmin_last - rmax_last)
    δ_hi = min(O + L - 1, cmax_last - rmin_last)

    for c_last in cmin_last:cmax_last
        δ_lo_c = max(δ_lo, c_last - rmax_last)
        δ_hi_c = min(δ_hi, c_last - rmin_last)
        active_c = max(0, δ_hi_c - δ_lo_c + 1)
        r_start_c = (c_last - δ_hi_c - rmin_last) * s_D
        cur, col_j = _pattern_nd!(
            rowval, colptr, offsets, row_rest, col_rest,
            Val(Nd), Val(Nd - 1),
            cur, col_j,
            row_base, active_c, r_start_c, s_D,
        )
    end

    return (cur, col_j)
end

# Method D: Nd ≥ 2, Nd ≠ D — non-D dim peel. Pure intersection on col_Nd
# vs row_Nd; for each valid c_Nd accumulate row_base and recurse.
@inline function _pattern_nd!(
    rowval::Vector{Int},
    colptr::Vector{Int},
    offsets::SUnitRange{O, L},
    row::NTuple{Nd, AbstractUnitRange{Int}},
    col::NTuple{Nd, AbstractUnitRange{Int}},
    ::Val{D}, ::Val{Nd},
    cur::Int, col_j::Int,
    row_base::Int, active::Int, r_start::Int, s_D::Int,
)::Tuple{Int, Int} where {O, L, D, Nd}
    row_last = last(row); col_last = last(col)
    row_rest = Base.front(row); col_rest = Base.front(col)
    rmin_last, rmax_last = first(row_last), last(row_last)
    cmin_last, cmax_last = first(col_last), last(col_last)

    s_Nd = prod(length, row_rest; init=1)
    cols_per_iter = prod(length, col_rest; init=1)

    for c_last in cmin_last:cmax_last
        if c_last >= rmin_last && c_last <= rmax_last
            new_row_base = row_base + (c_last - rmin_last) * s_Nd
            cur, col_j = _pattern_nd!(
                rowval, colptr, offsets, row_rest, col_rest,
                Val(D), Val(Nd - 1),
                cur, col_j,
                new_row_base, active, r_start, s_D,
            )
        else
            for _ in 1:cols_per_iter
                colptr[col_j + 1] = cur
                col_j += 1
            end
        end
    end

    return (cur, col_j)
end

"""
    _fill_nd!(nzval, offsets, term, row, col, ::Val{D}, ::Val{Nd},
              nzval_idx, active, k_hi, outer_coords) -> nzval_idx

Recursive N-D fill kernel. Mirrors `_pattern_nd!` structure; threads
`nzval_idx` (next free index in `nzval`) and `outer_coords` (mesh
coordinates of already-peeled dimensions, in the order matching coef
array axes) so coef indexing at the base fetches the column's `SVector`
once (`sv = term[c_1, outer_coords...]`) and reads slot `k`.

Threaded `active`, `k_hi` are set when peeling the stencil dim
(`Nd == D`), consumed at the inner intersection base.
"""
function _fill_nd! end

# Method A: (Nd, D) = (1, 1) — stencil base case.
@inline function _fill_nd!(
    nzval::AbstractVector{T},
    offsets::SUnitRange{O, L},
    term::AbstractArray{SVector{L, T}, N_coef},
    row::NTuple{1, AbstractUnitRange{Int}},
    col::NTuple{1, AbstractUnitRange{Int}},
    ::Val{1}, ::Val{1},
    nzval_idx::Int,
    ::Int, ::Int,
    outer_coords::NTuple{N_outer, Int},
)::Int where {O, L, T, N_coef, N_outer}
    rmin, rmax = first(row[1]), last(row[1])
    cmin, cmax = first(col[1]), last(col[1])

    δ_lo = max(O,         cmin - rmax)
    δ_hi = min(O + L - 1, cmax - rmin)
    δ_lo > δ_hi && return nzval_idx

    for c in cmin:cmax
        δ_lo_c = max(δ_lo, c - rmax)
        δ_hi_c = min(δ_hi, c - rmin)
        if δ_lo_c <= δ_hi_c
            sv = term[c, outer_coords...]
            for δ in δ_hi_c:-1:δ_lo_c
                k = δ - O + 1
                nzval[nzval_idx] = sv[k]
                nzval_idx += 1
            end
        end
    end
    return nzval_idx
end

# Method B: (Nd, D) = (1, D>1) — inner intersection base case.
@inline function _fill_nd!(
    nzval::AbstractVector{T},
    offsets::SUnitRange{O, L},
    term::AbstractArray{SVector{L, T}, N_coef},
    row::NTuple{1, AbstractUnitRange{Int}},
    col::NTuple{1, AbstractUnitRange{Int}},
    ::Val{D}, ::Val{1},
    nzval_idx::Int,
    active::Int, k_hi::Int,
    outer_coords::NTuple{N_outer, Int},
)::Int where {O, L, T, N_coef, D, N_outer}
    rmin, rmax = first(row[1]), last(row[1])
    cmin, cmax = first(col[1]), last(col[1])

    for c in cmin:cmax
        if c >= rmin && c <= rmax && active > 0
            sv = term[c, outer_coords...]
            for i in 0:(active - 1)
                k = k_hi - i
                nzval[nzval_idx] = sv[k]
                nzval_idx += 1
            end
        end
    end
    return nzval_idx
end

# Method C: Nd ≥ 2, Nd == D — stencil dim peel.
@inline function _fill_nd!(
    nzval::AbstractVector{T},
    offsets::SUnitRange{O, L},
    term::AbstractArray{SVector{L, T}, N_coef},
    row::NTuple{Nd, AbstractUnitRange{Int}},
    col::NTuple{Nd, AbstractUnitRange{Int}},
    ::Val{Nd}, ::Val{Nd},
    nzval_idx::Int,
    ::Int, ::Int,
    outer_coords::NTuple{N_outer, Int},
)::Int where {O, L, T, N_coef, Nd, N_outer}
    row_last = last(row); col_last = last(col)
    row_rest = Base.front(row); col_rest = Base.front(col)
    rmin_last, rmax_last = first(row_last), last(row_last)
    cmin_last, cmax_last = first(col_last), last(col_last)

    δ_lo = max(O,         cmin_last - rmax_last)
    δ_hi = min(O + L - 1, cmax_last - rmin_last)

    for c_last in cmin_last:cmax_last
        δ_lo_c = max(δ_lo, c_last - rmax_last)
        δ_hi_c = min(δ_hi, c_last - rmin_last)
        active_c = max(0, δ_hi_c - δ_lo_c + 1)
        k_hi_c = δ_hi_c - O + 1
        nzval_idx = _fill_nd!(
            nzval, offsets, term, row_rest, col_rest,
            Val(Nd), Val(Nd - 1),
            nzval_idx, active_c, k_hi_c,
            (c_last, outer_coords...),
        )
    end
    return nzval_idx
end

# Method D: Nd ≥ 2, Nd ≠ D — non-D dim peel.
@inline function _fill_nd!(
    nzval::AbstractVector{T},
    offsets::SUnitRange{O, L},
    term::AbstractArray{SVector{L, T}, N_coef},
    row::NTuple{Nd, AbstractUnitRange{Int}},
    col::NTuple{Nd, AbstractUnitRange{Int}},
    ::Val{D}, ::Val{Nd},
    nzval_idx::Int,
    active::Int, k_hi::Int,
    outer_coords::NTuple{N_outer, Int},
)::Int where {O, L, T, N_coef, D, Nd, N_outer}
    row_last = last(row); col_last = last(col)
    row_rest = Base.front(row); col_rest = Base.front(col)
    rmin_last, rmax_last = first(row_last), last(row_last)
    cmin_last, cmax_last = first(col_last), last(col_last)

    for c_last in cmin_last:cmax_last
        if c_last >= rmin_last && c_last <= rmax_last
            nzval_idx = _fill_nd!(
                nzval, offsets, term, row_rest, col_rest,
                Val(D), Val(Nd - 1),
                nzval_idx, active, k_hi,
                (c_last, outer_coords...),
            )
        end
    end
    return nzval_idx
end
