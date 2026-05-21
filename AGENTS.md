# AGENTS.md

CSC **assembly** invariants for the stencil types. The stencil/term *type
vocabulary* (`AccessStyle`, `AbstractStencil`, `AbstractTerm`,
`StaticPair`/`StaticShift`, `LinearStencil`, `StarStencil`, `Stencil`,
`as_linear`/`as_star`) is owned by **StencilCore** —
see [`../StencilCore/AGENTS.md`](../StencilCore/AGENTS.md). The symbolic CAS
that feeds these stencils is [`docs/cas.md`](docs/cas.md). Status:
[`docs/plan.md`](docs/plan.md); per-feature: [`docs/star.md`](docs/star.md).

This package depends on StencilCore (via a `[sources]` path until it is
registered), re-exports the stencil names, and adds `assemble` / `update!` /
`build`.

## Sticky decisions

1. **Public ops.** `assemble` (sparsity only — `colptr`/`rowval`, uninit
   `nzval`); `update!` (writes `nzval` in place, allocation-free modulo the
   single `term[c]` `getindex` per active column — `SVector{L}` is isbits,
   returned by value); `build = update!(assemble(...), ...)`.

2. **Dispatch gate.** Assembly methods require a **concrete-array**
   coefficient (`A<:AbstractArray{SVector{L,T},N}`) and `S = ColumnAccess`.
   A symbolic (`AbstractTerm`) coefficient or a `RowAccess` stencil matches
   no method → `MethodError` (materialize first / CSR is reserved). The
   scalar `T` is recovered as `eltype(E)`; the grid rank `N` is bound by
   unifying `A`'s `ndims` with `row::NTuple{N}`.

3. **Guards.** 1-D `LinearStencil` `assemble` / `update!` pin `D = 1`,
   `N = 1` (misuse → `MethodError`) and enforce
   **`L − 1 ≤ length(row[1])`** — the three-phase kernel's exact
   correctness boundary. N-D `LinearStencil` carries the same guard on
   `row[D]`; `StarStencil` requires **`2L ≤ length(row[d])` for every `d`**
   (also what makes the interlaced reverse-lex order coincide with CSC row
   order — see decision 6).

4. **Row / col.** `NTuple{N, AbstractUnitRange{Int}}` on a single shared
   integer mesh — rectangular sub-blocks, possibly unequal or shifted.
   Compact row index for mesh `r` is `r − first(row[d]) + 1` per dim.
   Column `c` × offset `δ` lands on row `c − δ`; out-of-range rows dropped,
   off-`col` columns never visited. Under `ColumnAccess`, `term[c_idx]` is
   read at the **column** mesh position; the coef array's axes must cover
   `col` (wrap shifted/non-square operators with `OffsetArray` or build in
   mesh-space). The whole `SVector` for a column is fetched in one
   `getindex`, so a lazy coef array can precompute per-column work.

5. **Three-phase 1-D `LinearStencil` kernel** (`_pattern!` / `_fill!`).
   Trim `δ_lo = max(O, cmin − rmax)`, `δ_hi = min(O+L−1, cmax − rmin)`;
   empty early if `δ_lo > δ_hi`. Otherwise tile `[cmin, cmax]`: left ramp
   (active grows to `Leff = δ_hi − δ_lo + 1`), interior (closed-form
   `cur(c) = cur_int_0 + Leff·(c − c_LR)`), right ramp (active shrinks).
   Each phase walks `δ` high → low so rows ascend per column (CSC
   sortedness without sort); `δ → k = δ − O + 1` indexes the column's
   `SVector`. Ramps use `max(0, active)` for off-mesh tails. Under the
   guard, interior empties at `L = length(row)+1`; the saturated-middle
   phase is out of scope.

6. **N-D dispatch.** State-threading recursion peels dimensions outermost
   (last) → innermost (first) via `Base.front` / `last`, dispatches on
   `Val{D}` × `Val{Nd}`, and **returns** updated state so each output column
   is visited once. The three-phase trim runs at the stencil-dim peel
   (`Nd == D`), threading `(active_c, r_start_c)` to the inner base, which
   emits an arithmetic sequence of step `s_D = prod(length(row[d]) for d in
   1:D−1)`; descending `δ` ⇒ ascending row. Non-D peels accumulate
   `row_base`; fill threads `outer_coords::NTuple` for dimension-agnostic
   `term[c_1, outer_coords...][k]` indexing.

7. **Interlaced `StarStencil` kernel** (`_pattern_nd_star!` /
   `_fill_nd_star!`). Per output column (column-major via
   `CartesianIndices(col)`) walk the `M = 2NL+1` canonical offsets in
   CSC-ascending-row order — upper block `d = N…1, o = L…1`, the explicit
   **diagonal slot**, lower block `d = 1…N, o = 1…L` — skipping off-mesh
   slots; row `= base − o·s_d`. Under the `2L ≤ length(row[d])` guard the
   reverse-lex storage order is strictly monotonic in row, so emission is
   sort-free. The **diagonal is a single slot** `term[c][NL+1]` (no
   per-axis merge). A column with one axis outside `row` emits only that
   axis's slots; with two or more, nothing. The 1-D star delegates to
   `LinearStencil{1}` via `_as_linear` (layouts coincide); `update!` is
   allocation-free.

## Public surface

Re-exports from StencilCore: `AccessStyle`, `ColumnAccess`, `RowAccess`,
`AbstractStencil`, `LinearStencil`, `StarStencil`, `Stencil`, `as_linear`,
`as_star`. Adds: `assemble`, `update!`, `build`.

Kernels (internal): `_pattern!` / `_fill!` (1-D LinearStencil),
`_pattern_nd!` / `_fill_nd!` (N-D LinearStencil),
`_pattern_nd_star!` / `_fill_nd_star!` (interlaced N-D StarStencil),
`_as_linear` (1-D star → LinearStencil).

Tests: `julia --project=. -e 'using Pkg; Pkg.test()'`.

## Scope

Implemented: CSC assembly for `LinearStencil` (any `1 ≤ D ≤ N`) and the
interlaced `StarStencil` (any `N ≥ 1`, `L ≥ 1`) under `S = ColumnAccess`,
with the per-dim guards above. Symbolic-coefficient and `RowAccess`
stencils construct but are unassemblable here by design.

Deferred: CSR assembly (`RowAccess`); a direct kernel for the general
`Stencil` (narrowed to Linear/Star for now); `BandedMatrix` / dense targets;
stencil composition.
