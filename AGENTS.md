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
names: `_pattern!`, `_fill!` (specialised fast path). N-D kernels:
`_pattern_nd!`, `_fill_nd!` — state-threading recursion that peels
dimensions outermost (last) → innermost (first) via `Base.front` /
`last`, dispatches on `Val{D}` × `Val{Nd}`, and **returns** updated
state (`(cur, col_j)` for pattern, `nzval_idx` for fill) so each output
column is visited exactly once. The three-phase trim runs at the
stencil-dim peel (`Nd == D`), yielding `(active_c, r_start_c)` that
are threaded to the inner base case which emits an arithmetic
sequence of step `s_D = prod(length(row[d]) for d in 1:D-1)`. Outer
non-D peels accumulate `row_base`; fill threads outer mesh coords as
an `NTuple` for dimension-agnostic `coefs[k][c_1, outer_coords...]`
indexing. Tests: `julia --project=. -e 'using Pkg; Pkg.test()'`.

## Scope

Implemented: `LinearStencil{D, O, L, T, N, C}` (any `1 ≤ D ≤ N`); 1-D
`assemble` / `update!` / `build` with the `L − 1 ≤ length(row[1])`
guard; N-D `assemble` / `update!` / `build` (2-D and 3-D tested for
all `D`) with the `L − 1 ≤ length(row[D])` guard. Deferred:
composition, `BandedMatrix` target, dense.
