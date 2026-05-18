# AGENTS.md

Canonical record of design decisions. See [`README.md`](README.md) for
quickstart, [`docs/plan.md`](docs/plan.md) for status and roadmap.

## Sticky decisions

1. **Type-driven API.**
   `LinearStencil{D, O, L, T, N, C<:NTuple{L, AbstractArray{T, N}}}` with
   `offsets::SUnitRange{O, L}` (`StaticArrays.jl`, ascending) and `coefs::C`
   in ascending offset order (`coefs[1]` ↦ offset `O`). `D` ∈ `[1, N]`;
   `O = Δ_min`, `L = Δ_max − Δ_min + 1`. Heterogeneous coef containers OK
   if `eltype`/`ndims` agree.

2. **Constructor.** Inner checks `D ≥ 1`, `D ≤ N`; `SUnitRange` enforces
   unit-ascending offsets at the type. Outer catch-all raises
   `ArgumentError` on ill-typed inputs.

3. **Public ops.** `assemble` (sparsity only, uninit `nzval`); `update!`
   (writes `nzval` in place, allocation-free modulo `coefs[k]` `getindex`);
   `build = update!(assemble(...), ...)`. 1-D `assemble` / `update!` pin
   `D = 1` and `N = 1` (misuse → `MethodError`) and enforce
   **`L − 1 ≤ length(row[1])`** at runtime — the three-phase kernel's
   exact correctness boundary.

4. **Row/col.** `NTuple{N, AbstractUnitRange{Int}}` on a single shared
   integer mesh — rectangular sub-blocks that may overlap arbitrarily
   and be unequal or shifted. Compact row index for mesh `r` is
   `r − first(row[d]) + 1` per dim.

5. **Coefs anchor.** `coefs[k][c_idx]` with `k = δ − O + 1`. Coef axes
   must cover `col`. For shifted / non-square operators, wrap with
   `OffsetArrays.OffsetArray` or build in mesh-space.

6. **Subtraction.** Column `c` (mesh `p_c`) × offset `δ` → row `p_c − δ`;
   out-of-range rows dropped, off-`col` columns never visited.

7. **Kernel — three-phase contiguous walk.** Trim
   `δ_lo = max(O, cmin − rmax)`, `δ_hi = min(O + L − 1, cmax − rmin)`;
   empty (early return) if `δ_lo > δ_hi`. Otherwise three phases tile
   `[cmin, cmax]`: left ramp (active grows to `Leff = δ_hi − δ_lo + 1`),
   interior (active const., closed-form
   `cur(c) = cur_int_0 + Leff * (c − c_LR)` ⇒ column writes
   independent), right ramp (active shrinks). Each phase walks `δ` high
   → low so rows ascend per column (CSC sortedness without sort);
   `δ → k = δ − O + 1` indexes `coefs`. Ramps use `max(0, active)` to
   absorb off-mesh column tails. Under the guard, interior empties at
   `L = length(row) + 1` and the ramps tile without gap; the
   saturated-middle phase is out of scope. See `?_pattern!` for
   per-phase formulas.

Public API: `LinearStencil`, `assemble`, `update!`, `build`. 1-D kernel
names: `_pattern!`, `_fill!`. N-D (deferred): tuple-length dispatch via
`Base.front` / `last`; accumulators as `Tuple`s, no `Ref`s. Tests:
`julia --project=. -e 'using Pkg; Pkg.test()'`.

## Scope

Implemented: `LinearStencil{D, O, L, T, N, C}` (any `1 ≤ D ≤ N`); 1-D
`assemble` / `update!` / `build` with the `L − 1 ≤ length(row[1])`
guard. Next milestone ([`docs/plan.md`](docs/plan.md)): N-D dispatch
via recursive dimensional-peeling kernels (tuple-length dispatch).
Deferred: composition, `BandedMatrix` target, dense.
