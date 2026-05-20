# AGENTS.md

Canonical record of design decisions. See [`README.md`](README.md) for
quickstart, [`docs/plan.md`](docs/plan.md) for status, and
[`docs/star.md`](docs/star.md) / [`docs/term.md`](docs/term.md) for
per-feature plans.

## Sticky decisions

1. **Type-driven API.**
   `LinearStencil{D, O, L, T, N, A<:AbstractArray{SVector{L, T}, N}, S<:AccessStyle} <: AbstractStencil{S}`
   carries `offsets::SUnitRange{O, L}` (`StaticArrays.jl`) and a single
   coefficient array `term::A`. Each element `term[c]` is the
   `SVector{L, T}` of the column's per-offset coefficients in ascending
   offset order (`term[c][1]` ↦ offset `O`) — an array-of-structs, not
   one array per offset. `T` is the scalar eltype; the `SVector` length
   is tied to the offset count `L` at the type level.
   `D` ∈ `[1, N]`; `O = δ_min`, `L = δ_max − δ_min + 1`. Offsets are
   **diagonal indices**: for column `j`, row `i`, the diagonal is
   `k = j − i`.
   `StarStencil{L, T, N, M, C<:NTuple{N, AbstractArray{SVector{M, T}, N}}, S} <: AbstractStencil{S}`
   is the star-shaped analog with `M = 2L + 1` per-axis offsets in
   `−L:L`; `terms[d]` is one `AbstractArray{SVector{M, T}, N}` per axis.

2. **Constructor.** Inner ctors check `D ≥ 1`, `D ≤ N` (LinearStencil)
   and `L ≥ 1`, `M = 2L + 1` (StarStencil); `SUnitRange` enforces
   unit-ascending offsets at the type, and the `SVector` element length
   (`= L`, resp. `M`) is bound at the type — a mismatch fails dispatch.
   Friendly outer ctors raise `ArgumentError` on ill-typed inputs
   (non-array `term`, non-`SVector` eltype, `SVector` length ≠ `L`/`M`).

3. **Public ops.** `assemble` (sparsity only, uninit `nzval`);
   `update!` (writes `nzval` in place, allocation-free modulo the single
   `term[c]` `getindex` per active column — `SVector{L}` is isbits,
   returned by value); `build = update!(assemble(...), ...)`.
   1-D `LinearStencil` `assemble` / `update!` pin `D = 1`, `N = 1`
   (misuse → `MethodError`) and enforce
   **`L − 1 ≤ length(row[1])`** — the three-phase kernel's exact
   correctness boundary. N-D analogs carry the same per-stencil-dim
   guard; `StarStencil` requires `2L ≤ length(row[d])` for every `d`.

4. **Row / col.** `NTuple{N, AbstractUnitRange{Int}}` on a single
   shared integer mesh — rectangular sub-blocks that may be unequal
   or shifted. Compact row index for mesh `r` is
   `r − first(row[d]) + 1` per dim. Column `c` × offset `δ` lands on
   row `c − δ`; out-of-range rows dropped, off-`col` columns never
   visited.

5. **Terms anchor.** Under `S = ColumnAccess`,
   `term[c_idx][k]` is the coefficient at column `c_idx` for offset
   `δ = k − 1 + O` (`StarStencil`: `terms[d][c_idx][k]`, `δ = k − L − 1`).
   The coef array's axes must cover `col`. For shifted / non-square
   operators, wrap with `OffsetArray` or build in mesh-space. The whole
   `SVector` for a column is fetched in one `getindex`, so a lazy coef
   array can precompute work shared across a column's offsets.

6. **Three-phase kernel.** Trim `δ_lo = max(O, cmin − rmax)`,
   `δ_hi = min(O + L − 1, cmax − rmin)`; empty early if
   `δ_lo > δ_hi`. Otherwise tile `[cmin, cmax]` with: left ramp
   (active grows to `Leff = δ_hi − δ_lo + 1`), interior (closed-form
   `cur(c) = cur_int_0 + Leff * (c − c_LR)`), right ramp (active
   shrinks). Each phase walks `δ` high → low so rows ascend per
   column (CSC sortedness without sort); `δ → k = δ − O + 1` indexes
   the column's `SVector` (`sv = term[c]`; `sv[k]`). Ramps use
   `max(0, active)` for off-mesh tails. Under the
   guard, interior empties at `L = length(row) + 1`; the
   saturated-middle phase is out of scope. See `?_pattern!`.

7. **N-D dispatch.** State-threading recursion peels dimensions
   outermost (last) → innermost (first) via `Base.front` / `last`,
   dispatches on `Val{D}` × `Val{Nd}`, and **returns** updated state
   (`(cur, col_j)` / `nzval_idx`) so each output column is visited
   once. The three-phase trim runs at the stencil-dim peel
   (`Nd == D`), threading `(active_c, r_start_c)` to the inner base,
   which emits an arithmetic sequence of step
   `s_D = prod(length(row[d]) for d in 1:D-1)`. Non-D peels
   accumulate `row_base`; fill threads `outer_coords::NTuple` for
   dimension-agnostic `term[c_1, outer_coords...][k]` indexing.
   `StarStencil` kernels (`_pattern_nd_star!`, `_fill_nd_star!`)
   follow the same peel structure with three-way per-column
   branching and a single merged CSC entry on the diagonal; the
   full-star fill fetches each axis's column `SVector` once
   (`svs = ntuple(d -> terms[d][c_1, …], N)`) and reuses it across that
   axis's offsets and the merged center sum `Σ_d svs[d][L + 1]`.

8. **Access style.** `AccessStyle` is a Holy trait carried as the
   **last** stencil type parameter; `AbstractStencil{S<:AccessStyle}`
   defines the accessor once. `ColumnAccess` (CSC) and `RowAccess`
   (reserved for CSR) are the singletons. Trait is inert at
   element-access time; assemblers dispatch on it. Construction uses
   a positional `Type` tag with `ColumnAccess` as default:
   `LinearStencil{D}(offsets, term)` ≡
   `LinearStencil{D}(ColumnAccess, offsets, term)`;
   `LinearStencil{D}(RowAccess, offsets, term)` is explicit. Same
   for `StarStencil{L}(…)`. CSC `assemble` / `update!` dispatch
   requires `S = ColumnAccess`; `RowAccess` → `MethodError` until a
   CSR assembler is added.

## Public surface

Exports: `LinearStencil`, `StarStencil`, `assemble`, `update!`,
`build`, `AbstractStencil`, `AccessStyle`, `ColumnAccess`,
`RowAccess`.

Kernels (internal): `_pattern!` / `_fill!` (1-D LinearStencil fast
path), `_pattern_nd!` / `_fill_nd!` (N-D LinearStencil),
`_pattern_nd_star!` / `_fill_nd_star!` (N-D StarStencil).

Tests: `julia --project=. -e 'using Pkg; Pkg.test()'`.

## Scope

Implemented: `LinearStencil` (any `1 ≤ D ≤ N`) and `StarStencil`
(any `N ≥ 1`, `L ≥ 1`) under `S = ColumnAccess`; full CSC pipeline
with the per-dim guards above. `RowAccess` constructs but is
unassemblable on CSC by design.

Deferred: CSR assembly (`RowAccess` activation), composition,
`BandedMatrix` and dense matrix targets.
