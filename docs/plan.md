# CartesianOperators implementation plan

Forward-looking. Reflects the move from arbitrary `NTuple{K, Int}`
offsets to contiguous `SUnitRange{O, L}` and the rewrite of the 1-D
kernel from event-merge segment walk to the three-phase shape.
Historical: [`docs/superpowers/specs/2026-05-12-cartesian-operators-design.md`](superpowers/specs/2026-05-12-cartesian-operators-design.md),
[`docs/superpowers/plans/2026-05-12-phase0-phase1.md`](superpowers/plans/2026-05-12-phase0-phase1.md)
(superseded).

## Status

**Implemented** (canonical spec in [`AGENTS.md`](../AGENTS.md)):

- `LinearStencil{D, O, L, T, N, A<:AbstractArray{SVector{L, T}, N}, S<:AccessStyle}`
  `<: AbstractStencil{S}` with contiguous unit-stride offsets
  (`SUnitRange{O, L}`) and a single coefficient array `term` whose
  elements are per-column `SVector{L}`s in ascending offset order.
- `StarStencil{L, T, N, M, C<:NTuple{N, AbstractArray{SVector{M, T}, N}}, S}`
  (`<: AbstractStencil{S}`) — N-D star-shaped operator with per-axis
  reach `−L … +L`, diagonal entries merged across axes
  (`A[r, r] = Σ_d terms[d][c][L + 1]`). See [`docs/star.md`](star.md)
  for the design.
- `AccessStyle` Holy trait (`ColumnAccess` / `RowAccess`) +
  `AbstractStencil` supertype. Assemblers dispatch on `S = ColumnAccess`
  for CSC output; `RowAccess` reserved for future CSR. See
  [`docs/term.md`](term.md).
- 1-D / N-D `assemble` / `update!` / `build` for both stencils with the
  per-axis `L − 1 ≤ length(row[D])` (LinearStencil) and
  `2L ≤ length(row[d]) ∀ d` (StarStencil) guards.
- Three-phase contiguous kernel for LinearStencil: closed-form interior
  `cur(c)`, ramps with `max(0, active)` off-mesh clipping. One
  `resize!`; otherwise allocation-free. StarStencil kernels follow the
  same dimension-peeling pattern with a 3-way per-column branch.
- Test suite: constructors, direct kernels, integration vs the
  brute-force oracle (`sum(LinearStencils)` for StarStencil), edge
  cases, guards, `D ≤ N` invariant, access-style + supertype trait.

Deliberately out of scope: named convenience constants, scalar-coef
sugar, non-contiguous offsets, arbitrary-mask row/col, CSR assembly.

## Roadmap

### Next milestone — N-D dispatch

Extend `assemble` / `update!` to `LinearStencil{D, O, L, T, N}` against
`NTuple{N, AbstractUnitRange{Int}}` for any `1 ≤ D ≤ N`. The stencil
acts on mesh dim `D`; the other `N − 1` dims are "outer" with zero
shift.

**Status:** ✅ Done. 2-D `D ∈ {1, 2}` and 3-D `D ∈ {1, 2, 3}` all pass
against the oracle (equal, shifted, and unequal ranges). Design notes
preserved below for context.

#### Root cause and fix

The 1-D three-phase kernel processes all offsets for a column inside
one offset loop, then writes `colptr[c-cmin+2]` exactly once per
column. The current N-D recursion violates this: at `Nd == D` (stencil
on outermost), it loops over offsets and recurses with the same
`col_idx` per iteration — later offsets overwrite earlier `colptr`
slots.

The fix is to apply the three-phase trim **at the stencil dim during
peeling** to compute, for each `c_D ∈ col_D`, the per-column nnz count
`active(c_D)` and the smallest dim-D row contribution `r_start(c_D)`.
Recursion then carries `(active, r_start)` down to the base case, where
each valid inner column emits `active` rows as an arithmetic sequence
of step `s_D = prod(length(row[d]) for d in 1:D-1)`. Each output column
is visited exactly once.

Row index identity (column at mesh `c = (c_1, …, c_N)`, offset `δ_k`):

```
i_k = 1 + Σ_{d≠D} (c_d − rmin_d)·s_d + (c_D − δ_k − rmin_D)·s_D
```

Descending `δ` ⇒ ascending `i_k` by `s_D` per step → CSC sortedness
falls out without sort.

#### Kernel: `_pattern_nd!`

State-threaded recursion mirroring `CartesianRuns._build_fused!` —
each call **returns** `(new_cur, new_col_j)` rather than mutating
shared counters across sibling calls. Peels `last(row)` / `last(col)`
via `Base.front`; dispatches on `Val{Nd}`.

| Case | Behavior |
|---|---|
| `Nd ≥ 2`, `Nd ≠ D` (non-D dim) | For each `c_Nd ∈ col_Nd`: if `c_Nd ∈ row_Nd`, accumulate `row_base += (c_Nd − rmin_Nd)·s_Nd` and recurse with same `(active, r_start)`. Else mark sub-columns empty (`colptr[col_j+1] = cur; col_j += 1` for each). |
| `Nd ≥ 2`, `Nd == D` (stencil dim) | Three-phase trim on `col_Nd` vs `row_Nd`. For each `c_Nd`: `δ_lo_c = max(δ_lo, c_Nd − rmax_Nd)`, `δ_hi_c = min(δ_hi, c_Nd − rmin_Nd)`, `active_c = max(0, δ_hi_c − δ_lo_c + 1)`, `r_start_c = (c_Nd − δ_hi_c − rmin_Nd)·s_D`. Recurse with `(active_c, r_start_c)`. |
| `Nd == 1`, `D == 1` (base, stencil here) | Three-phase walk on `row[1]` vs `col[1]` with `row_base` added to each emitted row index; write `colptr[col_j+1] = cur` per column with global `col_j`. |
| `Nd == 1`, `D > 1` (base, inner intersection) | For each `c_1 ∈ col[1]`: if `c_1 ∈ row[1]` and `active > 0`, emit `active` rows as `1 + row_base + (c_1 − rmin_1) + r_start + i·s_D` for `i = 0..active-1`, then `colptr[col_j+1] = cur; col_j += 1`. |

Signature:

```julia
function _pattern_nd!(
    rowval::Vector{Int}, colptr::Vector{Int},
    offsets::SUnitRange{O, L},
    row::NTuple{Nd, AbstractUnitRange{Int}},
    col::NTuple{Nd, AbstractUnitRange{Int}},
    ::Val{D}, ::Val{Nd},
    cur::Int, col_j::Int,
    row_base::Int, active::Int, r_start::Int, s_D::Int,
)::Tuple{Int, Int} where {O, L, D, Nd}
```

#### Companion: `_fill_nd!`

Same case split and state threading; threads `nzval_idx` instead of
`(cur, col_j)` (colptr already built). At the base case, walks the
same offset sequence per valid column, fetches the column's `SVector`
once (`sv = term[c_1, outer_coords...]`), and writes
`nzval[nzval_idx + i] = sv[k]` with `k = δ − O + 1`. Outer mesh coords
are threaded as an `NTuple` built during non-D dim peeling, so coef
indexing is dimension-agnostic.

#### Implementation order

1. Rewrite `_pattern_nd!` → 2-D `D=1` and `D=2` `assemble` green.
2. Rewrite `_fill_nd!` → 2-D `D=1` and `D=2` `build` green.
3. Add 2-D shifted/unequal-range tests.
4. Add 3-D `D = 1, 2, 3` tests.

Verification: the existing `stencil_reference` oracle
(`test/reference.jl`) is already N-D; comparison via `J == ref` covers
`colptr`, `rowval`, and `nzval` jointly.

### Further milestones (sketched)

- **Composition / Laplacian.** Either sum-of-matrices (per-dim
  second-differences assembled and added) or a new
  `CompositeStencil{N,...}` walking all stencils together. Decide when
  consumer demand surfaces.
- **Other matrix targets.** Reintroduce the trailing `::Type{MT}` arg
  on `assemble` when a second format actually motivates it.
  `BandedMatrix` is natural — the three-phase interior emits a stride-1
  row run per column (the band). Dense for small / debug.
- **GPU.** Interior is already closed-form `cur(c)` (one thread per
  column maps naturally); ramps want closed-form (triangular sum) if
  inter-column parallelism on them becomes the bottleneck. Alternative:
  histogram + parallel prefix-sum + per-`(c, k)` emit (canonical CSC
  pattern). Deferred until N-D is in.

See [`AGENTS.md`](../AGENTS.md) for the canonical invariants (offset
ordering, CSC sortedness, `L − 1 ≤ length(row)` guard, `max(0, active)`
ramp clipping, N-D dispatch shape).
