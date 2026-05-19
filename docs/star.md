# Plan: `StarStencil` — N-D variable-coefficient Laplacian-shape operator

## Context

The package currently exposes `LinearStencil{D, …}`, an axis-aligned 1-D
stencil applied along mesh dimension `D` and embedded into the N-D
operator via state-threaded dimension peeling (`_pattern_nd!` /
`_fill_nd!` in `src/stencil.jl`). Building a true N-D Laplacian today
requires assembling N `LinearStencil`s and summing them — N CSC
allocations, N kernel walks, and post-merge CSC bookkeeping.

`StarStencil` packages the N axis-aligned 1-D actions into one struct
with a single fused assembly. Concretely it targets variable-coefficient
operators of the Laplacian shape

    ψ[i, j, …] = Σ_d (1/Δx_d²) · ( −φ[…, i_d−1, …] + 2 φ[…] − φ[…, i_d+1, …] )

generalised to symmetric reach `−L … +L` per axis. The decomposition
`star = Σ_d LinearStencil_d` remains valid and is the testing oracle;
the implementation produces an equivalent CSC in one pass, with the
N axes sharing a single CSC diagonal entry (`Σ_d coefs[d][center][c]`).

## Sticky decisions captured up front

These were settled during the planning interview — encoded here as fixed:

1. **Coefficient ndims:** `AbstractArray{T, N}` (not `M`). Each per-axis,
   per-offset coef array is sized by the **mesh**; we read at the column's
   mesh position. `M` is the offsets count, not an array rank.
2. **Diagonal merge:** one CSC entry per row, value `Σ_d coefs[d][L+1][c]`.
   Matches `A = sum(LinearStencils)` exactly after CSC dedup.
3. **No L = 0**, **no per-axis L_d**, **no separate 1-D kernel** — the
   1-D path is a thin wrapper that builds a `LinearStencil{1}` and
   delegates to existing `assemble` / `update!`.
4. **Symmetry is symbolic only**: offsets range over `−L … +L` (enforced
   by `M = 2L + 1` in the struct). Numerical values in `coefs[d][k]`
   are free to be asymmetric across `k`. The kernel treats each
   `coefs[d][k]` as an independent mesh-indexed array.
5. **Success bar:** matches `sum(LinearStencil)` oracle for 1-D / 2-D /
   3-D; allocation-free `update!`; CSC-sorted row indices by construction;
   single recursive kernel (no per-N specialisation).

## API

### Struct

```julia
struct StarStencil{L, T, N, M, C <: NTuple{N, NTuple{M, AbstractArray{T, N}}}}
    coefs::C

    function StarStencil{L}(coefs::NTuple{N, NTuple{M, AbstractArray{T, N}}}) where {L, T, N, M}
        L isa Int && L >= 1 || throw(ArgumentError(
            "stencil reach L must be a positive Int (got $L)"))
        M == 2L + 1 || throw(ArgumentError(
            "coefs inner tuple length must be 2L+1=$(2L+1) (got $M)"))
        N >= 1 || throw(ArgumentError("N must be ≥ 1"))
        new{L, T, N, M, typeof(coefs)}(coefs)
    end
end
```

Friendly outer constructor — same shape as `LinearStencil`'s:
report `ArgumentError` for length mismatch / mixed eltype / mixed ndims /
non-array elements. Re-uses the same diagnostic style.

### Public ops

- `assemble(st::StarStencil, row, col) -> SparseMatrixCSC{T, Int}` —
  builds `colptr` + `rowval`, allocates uninitialised `nzval`.
- `update!(mat, st::StarStencil, row, col)` — fills `nzval` in place,
  allocation-free apart from `coefs[d][k][...]` `getindex`.
- `build(st::StarStencil, row, col) = update!(assemble(st, row, col), st, row, col)`.

Export: add `StarStencil` to `src/CartesianOperators.jl`.

### 1-D specialisation (no parallel codepath)

```julia
_as_linear(st::StarStencil{L, T, 1}) where {L, T} =
    LinearStencil{1}(SUnitRange(-L, L), st.coefs[1])

assemble(st::StarStencil{L, T, 1}, row, col) where {L, T} =
    assemble(_as_linear(st), row, col)

update!(mat, st::StarStencil{L, T, 1}, row, col) where {L, T} =
    update!(mat, _as_linear(st), row, col)
```

## Invariants & guard

**Per-axis guard** (raised at `assemble` / `update!`):

    2L <= length(row[d])     for every d ∈ 1:N

This is the LinearStencil-per-axis guard (`L_linear − 1 ≤ length(row[D])`
with `L_linear = 2L + 1`). It also ensures the **cross-axis row-ordering
invariant** used by the kernel: under `L ≤ length(row[d]) − 1` for
d = 1 … N−1, axis-d's positive-side rows (`r(c) + s_d … r(c) + L s_d`)
finish before axis-(d+1)'s start (`r(c) + s_{d+1}` with
`s_{d+1} = length(row[d]) · s_d ≥ (L + 1) · s_d`). Both conditions
collapse to the single guard above for `L ≥ 1`.

Error message mirrors LinearStencil's "saturated-middle is out of scope".

## Per-column emission order (single column, full star)

Let `s_d = ∏_{e<d} length(row[e])`, `r(c) = row_base(c) + (c_1 − rmin_1) + 1`,
`δ_hi_d_c = min(L, c_d − rmin_d)`, `δ_lo_d_c = max(−L, c_d − rmax_d)`.

Rows ascend per column under the guard:

| Block             | Order                            | Rows emitted                             |
|-------------------|----------------------------------|------------------------------------------|
| Axis-N above      | `δ = δ_hi_N_c … max(1, δ_lo_N_c)`| `r(c) − δ s_N`                           |
| Axis-(N−1) above  | …                                | `r(c) − δ s_{N−1}`                       |
| …                 |                                  |                                          |
| Axis-1 above      | `δ = δ_hi_1_c … 1`               | `r(c) − δ`                               |
| Center            | one slot                         | `r(c)`                                   |
| Axis-1 below      | `δ = −1 … δ_lo_1_c`              | `r(c) − δ`                               |
| …                 |                                  |                                          |
| Axis-N below      | `δ = min(−1, δ_hi_N_c) … δ_lo_N_c`| `r(c) − δ s_N`                          |

Skipped axes / empty intervals contribute zero rows but keep the rest
in order. Per-column nnz = `Σ_d active_d + [center_valid]`.

## Per-column validity branching

For column `c = (c_1, …, c_N)` with `valid_d := c_d ∈ row[d]`:

| `count(¬valid_d)` | Emission                                          |
|-------------------|---------------------------------------------------|
| 0                 | Full star — all axes + center                     |
| 1 (at `d_*`)      | Axis-`d_*` only — no center, only axis-`d_*` offsets |
| ≥ 2               | Nothing                                           |

(Axis-`d` requires `valid_e ∀ e ≠ d`, since the d-axis row keeps c_e
in dim e ≠ d. Center requires `valid_d ∀ d`.)

## Worked 2-D shape

For `N = 2`, `L = 1` (the canonical Laplacian), full-star column:

    rows = [ r − s_2,   r − 1,   r,   r + 1,   r + s_2 ]
    vals = [ coefs[2][1][c],
             coefs[1][1][c],
             coefs[1][2][c] + coefs[2][2][c],    # diagonal merge
             coefs[1][3][c],
             coefs[2][3][c] ]

with `s_2 = length(row[1])`, `c = (c_1, c_2)`. Per-column work is
4L + 1 = 5 rows. Boundary columns drop a strict subset of the off-center
slots; the center either contributes a merged value (both axes valid)
or is absent (one axis out, the only-axis branch).

## Worked 3-D shape sanity-check

For `N = 3`, `L = 1`, full-star column with `s_2 = length(row[1])`,
`s_3 = length(row[1]) · length(row[2])`:

    rows = [ r − s_3,    # axis 3 above
             r − s_2,    # axis 2 above
             r − 1,      # axis 1 above
             r,          # diagonal merge of three centers
             r + 1,      # axis 1 below
             r + s_2,    # axis 2 below
             r + s_3 ]   # axis 3 below

Under the guard `2L ≤ length(row[d])`, we have
`s_3 ≥ length(row[2]) · s_2 ≥ (L + 1) · s_2 = 2 s_2` and
`s_2 ≥ length(row[1]) ≥ 2`, so the ordering `−s_3 < −s_2 < −1 < 0 < 1 < s_2 < s_3`
holds. ✓ The same argument scales to arbitrary `N`: each axis-d block
spans `[r − L s_d, r − s_d] ∪ [r + s_d, r + L s_d]`, and the next
axis-(d+1) block lives entirely outside that span. This is the structural
property that makes "concatenate per-axis blocks" suffice for CSC
sortedness — no inter-axis merge needed.

## Recursive kernel (N-D, single set of methods)

Mirrors `_pattern_nd!` / `_fill_nd!` but without the D-vs-non-D dispatch
(all dims are stencil dims). Peels outermost (last) → innermost (first)
via `Base.front` / `last`, dispatches on `Val{Nd}` (no `Val{D}`).

### State threaded

- `cur`, `col_j` (pattern) / `nzval_idx` (fill): usual CSC bookkeeping.
- `outer_coords::NTuple{N_outer, Int}`: mesh positions of already-peeled
  dims, prepended on each peel — same convention as `_fill_nd!` so coef
  reads at the base are `coefs[d][k][c_1, outer_coords...]`.
- `row_base::Int`: row-linear-index shift from valid outer dims, i.e.
  `Σ_{d peeled and valid_d} (c_d − rmin_d) · s_d`.
- `n_outer_invalid::Int` (0, 1, or `≥ 2` shortcut): how many peeled outer
  dims have `c_d ∉ row[d]`. Drives the per-column branching at base.
- `invalid_outer_d::Int`: identity of the single invalid outer dim
  (meaningful only when `n_outer_invalid == 1`); used at base to know
  which axis can still contribute.

Note we **do not thread per-axis `(δ_lo_d, δ_hi_d)`** — they are O(1)
to recompute at the base from `outer_coords[d]` and `row[d]`, avoiding
variable-arity tuples through the recursion.

### Methods

- **Base case `Val{1}`** — for each `c_1 ∈ col[1]`:
  1. Compute axis-1 trim from `c_1` and `row[1]`.
  2. Decide branch from `(n_outer_invalid, c_1 ∈ row[1])`:
     - `(0, true)`: full star — recompute axes 2…N trims from
       `outer_coords` / `row[2:N]`, emit per the table above.
     - `(0, false)`: axis-1 only — emit axis-1 above + below
       (no center).
     - `(1, true)`: axis-`invalid_outer_d` only — recompute its trim
       from `outer_coords[invalid_outer_d]`, emit its block (no center).
     - `(1, false)` or `≥ 2`: emit nothing.
  3. Advance `cur` / `col_j` (pattern), `nzval_idx` (fill); set
     `colptr[col_j + 1] = cur` (pattern).
- **Recursive case `Val{Nd}` (Nd ≥ 2)** — for each `c_Nd ∈ col[Nd]`:
  - If `c_Nd ∈ row[Nd]`: `row_base += (c_Nd − rmin_Nd) · s_Nd`,
    `n_outer_invalid` unchanged. Recurse with `(c_Nd, outer_coords...)`.
  - If `c_Nd ∉ row[Nd]`: `n_outer_invalid += 1`. If reaches 2: pad
    `colptr` for `prod(length, col_rest)` empty columns, no recurse.
    If equals 1: record `invalid_outer_d = Nd`, recurse with unchanged
    `row_base` and `(c_Nd, outer_coords...)`.

### Why CSC sortedness falls out

Within a single column the emission table is row-ascending by
construction (under the guard). Columns are visited in CSC order
(`col_j` increases monotonically, outermost peel iterates
`cmin_Nd … cmax_Nd` left-to-right). Each output column is visited
exactly once because each recursive call returns updated state and the
outer loop never revisits its iterate — same proof shape as
`_pattern_nd!`.

### Why `update!` is allocation-free

`_fill_nd_star!` mirrors `_pattern_nd_star!` shape; no `push!` (writes
into pre-allocated `nzval`), no intermediate arrays, no tuple
reallocation beyond `outer_coords` growing by one `Int` per peel
(stack-allocated). Coef reads are O(1) for `Vector` / `Fill` /
`OffsetArray` — same property as LinearStencil's fill.

### Per-axis nnz total (for `resize!(rowval, total)` in `assemble`)

```
total_axis_d(c_d ∈ col[d]) = min(L, c_d − rmin_d) − max(1, c_d − rmax_d) + 1   # above
                           + max(0, min(−1, c_d − rmin_d) − max(−L, c_d − rmax_d) + 1)  # below
```

`total_nnz` = sum over valid columns of (axis contributions + center),
factorisable over axes given the guard. Cleanest:

```
total = Σ_c [count_full_star(c) | count_single_axis(c) | 0]
```

computed inline at the start of `_pattern_nd_star!` via a separate
small recursion that does **not** touch `rowval` — same pattern as
LinearStencil's `total = Σ_δ overlap(δ)` precompute (the existing
`_pattern!` resizes `rowval` up front). Alternative: append-and-resize
via `push!`, matching `_pattern_nd!` Methods A/B/C/D; simpler but
allocates incrementally. Choose **the precompute-then-resize variant**
for parity with the 1-D fast path; the precompute reuses the same
recursion shape so it's not a duplicate kernel — just a
"count instead of emit" leaf.

## Files

- `src/stencil.jl`:
  - Add `struct StarStencil{L, T, N, M, C}` + inner / outer constructors
    after the existing `LinearStencil` block.
  - Add `_as_linear(::StarStencil{L, T, 1})`.
  - Add `_pattern_nd_star!` + `_fill_nd_star!` recursive kernel methods
    (`Val{1}` base, `Val{Nd}` recursive).
  - Add `assemble(::StarStencil, …)`, `update!(…, ::StarStencil, …)`.
  - `build(::StarStencil, …)` falls out of the existing generic
    `build(st, row, col)` (already polymorphic).
- `src/CartesianOperators.jl`: add `StarStencil` to the `export` list.
- `test/test_stencil.jl`: append `@testset "StarStencil …"` blocks
  (see below). Re-uses existing `stencil_reference` and the
  `using FillArrays, SparseArrays, StaticArrays: SUnitRange` imports.
- `test/oracle.jl`: optional — add a 2-D Laplacian smoke check matching
  the new docstring example.

No new dependencies. No changes to `LinearStencil`, `_pattern!`,
`_fill!`, `_pattern_nd!`, `_fill_nd!`, or `_row_stride`.

## Reused utilities

- `_row_stride(row, Val(d))` — already in `src/stencil.jl:312`; reused
  for `s_d`.
- LinearStencil's per-axis trim formulas (`δ_lo = max(O, cmin − rmax)`
  etc.) — same shape with `O = −L`, `L_linear = 2L + 1`.
- `stencil_reference` (`test/reference.jl:18`) — used inside the
  decomposition oracle to assemble each per-axis `LinearStencil` and
  sum them for the test.

## Tests (append to `test/test_stencil.jl`)

```julia
@testset "StarStencil constructor" begin
    @test_throws ArgumentError StarStencil{0}(((Fill(1.0, 5), Fill(1.0, 5), Fill(1.0, 5)),))  # L < 1
    @test_throws ArgumentError StarStencil{1}(((Fill(1.0, 5), Fill(1.0, 5)),))                 # M ≠ 2L+1
    # mixed eltype / ndims / non-array → friendly errors
end

@testset "StarStencil 1-D (delegates to LinearStencil)" begin
    row = (1:6,); col = (1:6,)
    n = 6
    # Laplacian-shape: −1, 2, −1.
    coefs = ((Fill(-1.0, n), Fill(2.0, n), Fill(-1.0, n)),)
    st = StarStencil{1}(coefs)
    J = build(st, row, col)
    # Oracle: single LinearStencil with the same offsets/coefs.
    ln = LinearStencil{1}(SUnitRange(-1, 1), coefs[1])
    @test J == build(ln, row, col)
end

@testset "StarStencil 2-D vs sum(LinearStencils)" begin
    row = (1:5, 1:4); col = (1:5, 1:4)
    n1 = length(row[1]); n2 = length(row[2])
    # Two axes, L = 1, Laplacian on a 5×4 mesh.
    coefs = (
        (Fill(-1.0, n1, n2), Fill(2.0, n1, n2), Fill(-1.0, n1, n2)),
        (Fill(-1.0, n1, n2), Fill(2.0, n1, n2), Fill(-1.0, n1, n2)),
    )
    st = StarStencil{1}(coefs)
    J  = build(st, row, col)
    L1 = build(LinearStencil{1}(SUnitRange(-1, 1), coefs[1]), row, col)
    L2 = build(LinearStencil{2}(SUnitRange(-1, 1), coefs[2]), row, col)
    @test J == L1 + L2
end

@testset "StarStencil 2-D with shifted col" begin
    # col ⊋ row in one axis to exercise the (n_outer_invalid == 1) branch.
    row = (1:5, 1:4); col = (1:5, 1:6)
    # ... same oracle: J == L1 + L2
end

@testset "StarStencil 3-D vs sum(LinearStencils)" begin
    row = (1:4, 1:3, 1:3); col = row
    # L = 1, three axes; same Fill(-1, 2, -1) per axis.
    # ... J == L1 + L2 + L3
end

@testset "StarStencil guard" begin
    @test_throws ArgumentError build(
        StarStencil{2}(((Fill(0.0, 3), Fill(0.0, 3), Fill(0.0, 3), Fill(0.0, 3), Fill(0.0, 3)),)),
        (1:3,), (1:3,),
    )  # 2L = 4 > length(row[1]) = 3
end
```

The 2-D Float32 / 3-D unequal-length variants follow the same recipe;
add them to mirror LinearStencil's coverage.

## Verification

1. Static: `julia --project=. -e 'using CartesianOperators'` loads without
   warning.
2. `julia --project=. -e 'using Pkg; Pkg.test()'` — all existing
   LinearStencil testsets remain green, new StarStencil testsets pass.
3. Allocation check (manual, not in the test suite): wrap the 2-D and
   3-D `update!` calls in `@allocated` after one warmup — expect 0
   bytes for `Fill` coefs.
4. Optional sanity run: `julia --project=. test/oracle.jl` if a 2-D
   Laplacian block is added there.

## Non-goals (explicit)

- L = 0 (degenerate diagonal). Caller can use a `LinearStencil{1}` with
  a single zero offset if they need that.
- Per-axis `L_d`. One scalar `L` shared across all axes.
- A bespoke 1-D `_star_*` kernel (1-D delegates to LinearStencil).
- Numerical value symmetry of any kind — `coefs[d][k]` arrays are
  treated as independent. Tests can deliberately use asymmetric
  values to strengthen the `sum(LinearStencil)` oracle.
- Composition with `LinearStencil` (`StarStencil + LinearStencil`,
  product, etc.) — out of scope; existing roadmap defers composition.
