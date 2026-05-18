"""
    LinearStencil{D, O, L, T, N, C<:NTuple{L, AbstractArray{T, N}}}

Variable-coefficient stencil with **contiguous** offsets, aligned with mesh
dimension `D`. For column `c` at mesh position `p_c` and offset `δ`, the
matrix entry lands on row `p_c − δ` with coefficient `coefs[δ − O + 1][p_c]`
— coefs are **column-anchored** and indexed in **ascending offset order**.

Type parameters: `D` is the mesh dim (1-based, `1 ≤ D ≤ N`); `O = Δ_min`
and `L = Δ_max − Δ_min + 1` are static via the `SUnitRange` type; `T`, `N`
are the shared coef `eltype` / `ndims`; `C` is the concrete coef-tuple
type. Heterogeneous coef containers (e.g. `Fill` + `Vector` +
`OffsetArray`) are fine if `eltype` / `ndims` agree.

Inner constructor `LinearStencil{D}(offsets::SUnitRange{O, L}, coefs)`
checks `D ≥ 1` and `D ≤ N`; unit-ascending offset order is forced by the
`SUnitRange` type. A catch-all outer constructor reports `ArgumentError`s
for non-`SUnitRange` offsets, length mismatch, mixed `eltype` / `ndims`.

# Example

```julia
using FillArrays, StaticArrays: SUnitRange
# Forward x-difference: offsets 0, 1; coefs ascending.
forward = LinearStencil{1}(SUnitRange(0, 1), (Fill(-1.0, 5), Fill(1.0, 5)))
J = build(forward, (1:5,), (1:5,))
```

See [`assemble`](@ref), [`update!`](@ref), [`build`](@ref).
"""
struct LinearStencil{D, O, L, T, N, C<:NTuple{L, AbstractArray{T, N}}}
    offsets::SUnitRange{O, L}
    coefs::C

    function LinearStencil{D}(
        offsets::SUnitRange{O, L},
        coefs::NTuple{L, AbstractArray{T, N}},
    ) where {D, O, L, T, N}
        D isa Int && D >= 1 || throw(ArgumentError(
            "stencil dimension D must be a positive Int (got $D)"))
        D <= N || throw(ArgumentError(
            "stencil dimension D=$D exceeds coef-array dimension N=$N"))
        new{D, O, L, T, N, typeof(coefs)}(offsets, coefs)
    end
end

# Friendly outer constructor: reports specific errors when the inner method's
# NTuple{L, AbstractArray{T, N}} signature does not match (length mismatch,
# non-array elements, mixed eltype, mixed ndims).
function LinearStencil{D}(offsets::SUnitRange{O, L}, coefs::Tuple) where {D, O, L}
    length(coefs) == L || throw(ArgumentError(
        "offsets has length $L but coefs has length $(length(coefs))"))
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

# Fallback for non-SUnitRange offsets — tells the user how to migrate.
function LinearStencil{D}(offsets, coefs) where {D}
    throw(ArgumentError(
        "offsets must be a StaticArrays.SUnitRange (contiguous unit-stride). " *
        "Got $(typeof(offsets)). Construct one via SUnitRange(Δ_min, Δ_max), " *
        "and supply coefs in ascending-offset order (coefs[1] is for offset Δ_min)."))
end

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
    _fill!(nzval, offsets::SUnitRange{O, L}, coefs, row, col)

1-D fill kernel — same three-phase shape as [`_pattern!`](@ref); writes
only `nzval`, never touches `colptr`. Indexes `coefs[k]` with
`k = δ − O + 1` (ascending offset order). Allocation-free apart from
`coefs[k]` `getindex` (O(1) for `Vector` / `Fill` / `OffsetArray`).
"""
function _fill!(
    nzval::AbstractVector{T},
    offsets::SUnitRange{O, L},
    coefs::NTuple{L, AbstractArray{T, 1}},
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
        for i in 0:(active - 1)
            δ = (c - rmin) - i
            k = δ - O + 1
            nzval[cur + i] = coefs[k][c]
        end
        cur += active
    end

    # Interior: closed-form cur(c); δ descends from δ_hi to δ_lo.
    cur_int_0 = cur
    for c in c_LR:c_RR
        cur_c = cur_int_0 + Leff * (c - c_LR)
        for i in 0:(Leff - 1)
            δ = δ_hi - i
            k = δ - O + 1
            nzval[cur_c + i] = coefs[k][c]
        end
    end
    cur = cur_int_0 + Leff * max(0, c_RR - c_LR + 1)

    # Right ramp: δ descends from δ_hi to c − rmax.
    for c in (c_RR + 1):cmax
        r0     = c - δ_hi - rmin + 1
        active = max(0, rmax - rmin + 1 - r0 + 1)
        for i in 0:(active - 1)
            δ = δ_hi - i
            k = δ - O + 1
            nzval[cur + i] = coefs[k][c]
        end
        cur += active
    end
    return
end

"""
    assemble(st::LinearStencil{1, O, L, T, 1},
             row::NTuple{1, AbstractUnitRange{Int}},
             col::NTuple{1, AbstractUnitRange{Int}}) -> SparseMatrixCSC{T, Int}

Build the sparsity pattern (`colptr`, `rowval`) of the operator induced by
`st` between `row` and `col` and allocate `nzval` **uninitialised**. Call
[`update!`](@ref) to populate `nzval`, or use [`build`](@ref) to do both
in one shot.

`row` and `col` are interpreted on a shared integer mesh; the entry at
column `c ∈ col[1]` and offset `δ` lands on row `c − δ` iff it lies in
`row[1]`.

Dispatch pins `D = 1` and `N = 1` (misuse → `MethodError`); the
`L − 1 ≤ length(row[1])` guard is the three-phase kernel's exact
correctness boundary.
"""
function assemble(
    st::LinearStencil{1, O, L, T, 1},
    row::NTuple{1, AbstractUnitRange{Int}},
    col::NTuple{1, AbstractUnitRange{Int}},
) where {O, L, T}
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
    update!(mat::SparseMatrixCSC{T, Int}, st::LinearStencil{1, O, L, T, 1},
            row::NTuple{1, AbstractUnitRange{Int}},
            col::NTuple{1, AbstractUnitRange{Int}}) -> mat

Write `mat.nzval` in place by re-walking `row`/`col` with `st`. `mat` must
have been produced by a matching [`assemble`](@ref) (same `st`, `row`,
`col`) so its `colptr`/`rowval` align with the kernel's sweep. Carries the
same `L − 1 ≤ length(row[1])` guard.
"""
function update!(
    mat::SparseMatrixCSC{T, Int},
    st::LinearStencil{1, O, L, T, 1},
    row::NTuple{1, AbstractUnitRange{Int}},
    col::NTuple{1, AbstractUnitRange{Int}},
) where {T, O, L}
    L - 1 <= length(row[1]) || throw(ArgumentError(
        "stencil width L=$L exceeds length(row[1])+1=$(length(row[1])+1); " *
        "the three-phase kernel is exact up to L = length(row) + 1, beyond which " *
        "the saturated-middle phase would be required and is out of scope"))
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

# N-D assembly via recursive dimensional peeling (Nd ≥ 2).
# Dispatches on tuple length; most-specific 1-D methods stay unchanged.

function assemble(
    st::LinearStencil{D, O, L, T, N},
    row::NTuple{N, AbstractUnitRange{Int}},
    col::NTuple{N, AbstractUnitRange{Int}},
) where {D, O, L, T, N}
    L - 1 <= length(row[D]) || throw(ArgumentError(
        "stencil width L=$L exceeds length(row[$D])+1=$(length(row[D])+1); " *
        "the three-phase kernel is exact up to L = length(row) + 1, beyond which " *
        "the saturated-middle phase would be required and is out of scope"))
    m, n = prod(length, row), prod(length, col)
    colptr = Vector{Int}(undef, n + 1); colptr[1] = 1
    rowval = Int[]
    _pattern_nd!(rowval, colptr, st.offsets, row, col, Val(D), Val(N))
    nzval = Vector{T}(undef, length(rowval))
    SparseMatrixCSC{T, Int}(m, n, colptr, rowval, nzval)
end

function update!(
    mat::SparseMatrixCSC{T, Int},
    st::LinearStencil{D, O, L, T, N},
    row::NTuple{N, AbstractUnitRange{Int}},
    col::NTuple{N, AbstractUnitRange{Int}},
) where {D, O, L, T, N}
    L - 1 <= length(row[D]) || throw(ArgumentError(
        "stencil width L=$L exceeds length(row[$D])+1=$(length(row[D])+1); " *
        "the three-phase kernel is exact up to L = length(row) + 1, beyond which " *
        "the saturated-middle phase would be required and is out of scope"))
    _fill_nd!(mat.nzval, st.offsets, st.coefs, row, col, Val(D), Val(N))
    return mat
end

"""
    _pattern_nd!(rowval, colptr, offsets, row, col, ::Val{D}, ::Val{Nd}) -> ()

Recursive N-D pattern kernel. Peels dimensions from outermost (last) to
innermost (first). At each level, branches on whether Nd == D (stencil
sweep) or Nd ≠ D (pure intersection). Base case (Nd=1) dispatches to
specialized 1-D handlers based on whether D == 1 or D > 1.
"""
function _pattern_nd! end

# Base case: Nd=1, D=1 (innermost dimension is the stencil dimension).
function _pattern_nd!(
    rowval::Vector{Int},
    colptr::Vector{Int},
    offsets::SUnitRange{O, L},
    row::NTuple{1, AbstractUnitRange{Int}},
    col::NTuple{1, AbstractUnitRange{Int}},
    ::Val{1},
    ::Val{1},
) where {O, L}
    # Use the existing 1-D kernel. colptr has size length(col)+1 and we fill it in place.
    _pattern!(rowval, colptr, offsets, row[1], col[1])
end

# Base case: Nd=1, D>1 (shouldn't happen; D ≤ N invariant prevents it,
# but include for completeness). This would be pure intersection.
function _pattern_nd!(
    rowval::Vector{Int},
    colptr::Vector{Int},
    offsets::SUnitRange{O, L},
    row::NTuple{1, AbstractUnitRange{Int}},
    col::NTuple{1, AbstractUnitRange{Int}},
    ::Val{D},
    ::Val{1},
) where {O, L, D}
    # Pure intersection: for each column in col[1], emit a row iff it's in row[1].
    rmin, rmax = first(row[1]), last(row[1])
    cmin, cmax = first(col[1]), last(col[1])
    cur = 1
    for c in cmin:cmax
        if c >= rmin && c <= rmax
            push!(rowval, c - rmin + 1)
            cur += 1
        end
        colptr[c - cmin + 2] = cur
    end
end

# Recursive case: Nd≥2, stencil dimension Nd==D (the peeled dim is the stencil dim).
# Use a helper function that the compiler can inline to branch on Nd==D at compile time.

# Recursive case: Nd≥2. Use two separate methods distinguished by Nd and D at compile time.
# Julia's type system will select the right one based on whether Nd==D.

function _pattern_nd!(
    rowval::Vector{Int},
    colptr::Vector{Int},
    offsets::SUnitRange{O, L},
    row::NTuple{Nd, AbstractUnitRange{Int}},
    col::NTuple{Nd, AbstractUnitRange{Int}},
    ::Val{D},
    ::Val{Nd},
) where {O, L, Nd, D}
    _pattern_nd_recursive(rowval, colptr, offsets, row, col, Val(D))
end

# Helper that dispatches based on tuple length at compile time.
# Threads global column index and outer-dimension column index through the recursion.
@inline function _pattern_nd_recursive(
    rowval::Vector{Int},
    colptr::Vector{Int},
    offsets::SUnitRange{O, L},
    row::NTuple{N_dims, AbstractUnitRange{Int}},
    col::NTuple{N_dims, AbstractUnitRange{Int}},
    ::Val{D},
    col_idx::Int = 1,
    outer_col_idx::Int = 1,
) where {O, L, N_dims, D}
    if N_dims == 1
        # Compute row_offset from outer_col_idx (column index in outer dimensions).
        # row_offset = (outer_col_idx - 1) * length(row[1])
        row_offset = (outer_col_idx - 1) * length(row[1])

        if D == 1
            # Base case: D=1 stencil on dimension 1.
            rmin, rmax = first(row[1]), last(row[1])
            cmin, cmax = first(col[1]), last(col[1])
            cur = colptr[col_idx]
            for c in cmin:cmax
                O_val = first(offsets)
                L_val = length(offsets)
                δ_lo = max(O_val, c - rmax)
                δ_hi = min(O_val + L_val - 1, c - rmin)
                if δ_lo <= δ_hi
                    for δ in δ_hi:-1:δ_lo
                        r_local = c - δ - rmin + 1
                        r_global = r_local + row_offset
                        push!(rowval, r_global)
                        cur += 1
                    end
                end
                colptr[col_idx + 1] = cur
                col_idx += 1
            end
        else
            # Base case: D > N_dims means stencil dimension was peeled earlier.
            # Still emit entries for all valid offsets.
            rmin, rmax = first(row[1]), last(row[1])
            cmin, cmax = first(col[1]), last(col[1])
            O_val = first(offsets)
            L_val = length(offsets)
            cur = colptr[col_idx]
            for c in cmin:cmax
                for k in 1:L_val
                    δ = O_val + k - 1
                    r = c - δ
                    if r >= rmin && r <= rmax
                        r_local = r - rmin + 1
                        r_global = r_local + row_offset
                        push!(rowval, r_global)
                        cur += 1
                    end
                end
                colptr[col_idx + 1] = cur
                col_idx += 1
            end
        end
    else
        # Nd ≥ 2: peel the last dimension.
        row_last = last(row)
        col_last = last(col)
        row_rest = Base.front(row)
        col_rest = Base.front(col)

        rmin_last, rmax_last = first(row_last), last(row_last)
        cmin_last, cmax_last = first(col_last), last(col_last)

        # Columns per iteration.
        cols_per_iter = prod(length.(col_rest); init=1)

        if N_dims == D
            # Stencil along the outermost dimension: just iterate over c_last.
            # The three-phase logic is applied in the base case (dim 1).
            for i_last in 1:(cmax_last - cmin_last + 1)
                _pattern_nd_recursive(rowval, colptr, offsets, row_rest, col_rest, Val(D), col_idx, i_last)
                col_idx += cols_per_iter
            end
        else
            # Pure intersection along the outermost dimension.
            for i_last in 1:(cmax_last - cmin_last + 1)
                _pattern_nd_recursive(rowval, colptr, offsets, row_rest, col_rest, Val(D), col_idx, i_last)
                col_idx += cols_per_iter
            end
        end
    end
end

"""
    _fill_nd!(nzval, offsets, coefs, row, col, ::Val{D}, ::Val{Nd}) -> ()

Recursive N-D fill kernel. Same structure as _pattern_nd! but writes nzval
using coefs indexed at column mesh positions.
"""
function _fill_nd!(
    nzval::AbstractVector{T},
    offsets::SUnitRange{O, L},
    coefs::NTuple{L, AbstractArray{T, N_coef}},
    row::NTuple{Nd, AbstractUnitRange{Int}},
    col::NTuple{Nd, AbstractUnitRange{Int}},
    ::Val{D},
    ::Val{Nd},
) where {O, L, T, N_coef, Nd, D}
    _fill_nd_recursive(nzval, offsets, coefs, row, col, Val(D))
end

@inline function _fill_nd_recursive(
    nzval::AbstractVector{T},
    offsets::SUnitRange{O, L},
    coefs::NTuple{L, AbstractArray{T, N_coef}},
    row::NTuple{N_dims, AbstractUnitRange{Int}},
    col::NTuple{N_dims, AbstractUnitRange{Int}},
    ::Val{D},
    col_idx::Int = 1,
    nzval_idx::Int = 1,
    outer_col_idx::Int = 1,
) where {O, L, T, N_coef, N_dims, D}
    if N_dims == 1
        # Convert col_idx and outer_col_idx to mesh coordinates for coef indexing.
        len1 = length(col[1])
        c1 = ((col_idx - 1) % len1) + 1

        if D == 1
            # Base case: fill nzval for the inner dimension with D=1 stencil.
            rmin, rmax = first(row[1]), last(row[1])
            cmin, cmax = first(col[1]), last(col[1])

            δ_lo = max(first(offsets), cmin - rmax)
            δ_hi = min(last(offsets), cmax - rmin)
            δ_lo > δ_hi && return (nzval_idx, col_idx)

            Leff = δ_hi - δ_lo + 1
            c_LR = max(cmin, rmin + δ_hi)
            c_RR = min(cmax, rmax + δ_lo)

            # Left ramp.
            for c in cmin:(c_LR - 1)
                active = max(0, c - rmin - δ_lo + 1)
                for i in 0:(active - 1)
                    δ = (c - rmin) - i
                    k = δ - first(offsets) + 1
                    # Index coefs with mesh coordinates: c1 (from inner dim) and outer_col_idx (from outer dims).
                    nzval[nzval_idx + i] = coefs[k][c1, outer_col_idx]
                end
                nzval_idx += active
                c1 += 1
            end

            # Interior.
            cur_int_0 = nzval_idx
            for c in c_LR:c_RR
                cur_c = cur_int_0 + Leff * (c - c_LR)
                for i in 0:(Leff - 1)
                    δ = δ_hi - i
                    k = δ - first(offsets) + 1
                    nzval[cur_c + i] = coefs[k][c1, outer_col_idx]
                end
                c1 += 1
            end
            nzval_idx = cur_int_0 + Leff * max(0, c_RR - c_LR + 1)

            # Right ramp.
            for c in (c_RR + 1):cmax
                r0 = c - δ_hi - rmin + 1
                active = max(0, rmax - rmin + 1 - r0 + 1)
                for i in 0:(active - 1)
                    δ = δ_hi - i
                    k = δ - first(offsets) + 1
                    nzval[nzval_idx + i] = coefs[k][c1, outer_col_idx]
                end
                nzval_idx += active
                c1 += 1
            end
            return (nzval_idx, col_idx)
        else
            # Base case: D > N_dims means stencil dimension was peeled earlier.
            # This dimension is an intersection, but we still emit entries for all valid offsets.
            rmin, rmax = first(row[1]), last(row[1])
            cmin, cmax = first(col[1]), last(col[1])
            O_val = first(offsets)
            L_val = length(offsets)

            for c in cmin:cmax
                # For each column c, emit entries for all valid offsets
                for k in 1:L_val
                    δ = O_val + k - 1
                    r = c - δ
                    if r >= rmin && r <= rmax
                        nzval[nzval_idx] = coefs[k][c, outer_col_idx]
                        nzval_idx += 1
                    end
                end
            end
            return (nzval_idx, col_idx)
        end
    else
        # Nd ≥ 2: peel the last dimension.
        row_last = last(row)
        col_last = last(col)
        row_rest = Base.front(row)
        col_rest = Base.front(col)

        rmin_last, rmax_last = first(row_last), last(row_last)
        cmin_last, cmax_last = first(col_last), last(col_last)

        # Columns per iteration.
        cols_per_iter = prod(length.(col_rest); init=1)

        if N_dims == D
            # Stencil along the outermost dimension: just iterate over c_last.
            for i_last in 1:(cmax_last - cmin_last + 1)
                nzval_idx, col_idx = _fill_nd_recursive(nzval, offsets, coefs, row_rest, col_rest, Val(D), col_idx, nzval_idx, i_last)
            end
        else
            # Pure intersection along the outermost dimension.
            for i_last in 1:(cmax_last - cmin_last + 1)
                nzval_idx, col_idx = _fill_nd_recursive(nzval, offsets, coefs, row_rest, col_rest, Val(D), col_idx, nzval_idx, i_last)
            end
        end
        return (nzval_idx, col_idx)
    end
end

