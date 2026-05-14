# CartesianOperators implementation plan

Forward-looking; updated after the LinearStencil refactor (commit `37bd7ae`).
For the "why" behind the abstractions, see
[`docs/superpowers/specs/2026-05-12-cartesian-operators-design.md`](superpowers/specs/2026-05-12-cartesian-operators-design.md).
The original task-by-task plan at
[`docs/superpowers/plans/2026-05-12-phase0-phase1.md`](superpowers/plans/2026-05-12-phase0-phase1.md)
is historical; it was overtaken by the move to a type-driven API.

## Status

Implemented:

- `LinearStencil{D,K,T}` — constant-coefficient 1-D stencil aligned with
  mesh dimension `D`. Inner constructor validates `D ≥ 1` and strict-
  descending offsets via `issorted(offsets; lt = >=)`.
- `assemble(st, row, col, ::Type{SparseMatrixCSC{T,Int}} = …)` and
  `update!(mat, st, row, col)` for `LinearStencil{1}` against
  `CartesianRunIndices{1}`.
- Pointer-sweep kernels `_pattern_runs!` and `_fill_runs!` (1-D base; do
  not call `Base.in` or `getindex` on the cri).
- Test suite (23 assertions): constructor validation, 1-D kernel direct
  tests, integration tests for forward / backward / central
  x-differences against `stencil_reference`, Float32 element type, and
  `D ≠ 1` dispatch rejection.
- Cross-check oracle in `test/oracle.jl`: three independent correctness
  routes (pointer-sweep kernel via `assemble`+`update!`, dictionary-based
  `stencil_reference`, `spdiagm`-based `stencil_naive_x`) agree on random
  masks.

Deliberately not exported / not shipped:

- Named convenience constants (`forward_x`, etc.). Callers construct
  `LinearStencil{1}((1, 0), (1.0, -1.0))` inline.
- `domain(cri)` access. The package treats `CartesianRunIndices` as
  carrying only `intervals`, `offsets`, and the `length`/iteration
  interface (see `AGENTS.md`).

## Roadmap

### Next milestone — N-D dispatch

Extend `assemble` / `update!` to `LinearStencil{D}` against
`CartesianRunIndices{N}` for any `1 ≤ D ≤ N`. The stencil acts on mesh
dim `D`; the remaining `N − 1` dims are "outer" with zero shift. The
same code path handles every D — `D = 1` is x-aligned, `D = 2`
y-aligned, and so on; the original Phase 1b (N-D x) and Phase 2 (y in
N-D) collapse into one piece of work.

#### Kernel structure

Two new recursive kernels mirror the dispatch style of CartesianRuns's
`_intersect_fused!`. They take the `LinearStencil` directly (it carries
`D` as a type parameter plus the offsets/coefs the kernels need — see
"Decisions made" below):

```
_pattern_fused!(rowval, colptr, st::LinearStencil{D},
                row_ivs::NTuple{Nd}, row_offs, row_lo, row_hi,
                col_ivs::NTuple{Nd}, col_offs, col_lo, col_hi)

_fill_fused!(nzval, colptr, st::LinearStencil{D}, coef_selected,
             row_ivs::NTuple{Nd}, row_offs, row_lo, row_hi,
             col_ivs::NTuple{Nd}, col_offs, col_lo, col_hi)
```

At each recursion level, dispatch on `Nd` (current dim, equal to the
tuple length) vs `D` (stencil dim from `st`'s type parameter):

| `Nd` vs `D`           | Action                                                                 |
|-----------------------|------------------------------------------------------------------------|
| `Nd > D`              | Outer dim, intersection sweep (zero shift). Peel and recurse.          |
| `Nd == D` (`Nd ≥ 2`)  | Stencil-applying outer sweep. For each offset `k`, peel and recurse.   |
| `Nd < D` (`Nd ≥ 2`)   | Inner dim (only reached when `D > 1`). Intersection sweep. Recurse.    |
| `Nd == 1`             | Base case (see below).                                                 |

The "intersection outer sweep" is structurally identical to
CartesianRuns's `_intersect_fused!` recursive case (two-pointer sweep
on the dim-`Nd` interval vectors of row and col, computing inner slices
via `_inner_slice`). The "stencil-applying outer sweep" is the same
shape with an explicit per-offset pointer into the row-side intervals,
shifted by `offsets[k]` in dim `Nd`.

**Recursion order is load-bearing.** The kernels must peel the
*outermost* dimension first (dim `N`, then `N-1`, …, down to dim `1` at
the base case), so the innermost dim — which varies fastest in
compact-column order — is enumerated last. This is what makes columns
come out in compact-ascending order across the whole recursion, which
in turn lets `colptr` be filled incrementally (`colptr[c+1] =
colptr[c] + cnt`) without a post-hoc sort. Reversing the peel direction
would silently break CSC's column ordering. This mirrors CartesianRuns's
`_intersect_fused!`, which peels `last(intervals)` first.

**`_inner_slice` is internal to CartesianRuns** (underscore-prefixed,
not exported). The module file needs an explicit
`using CartesianRuns: _inner_slice` alongside the existing
`using CartesianRuns: Interval, shift`.

#### Base case (`Nd == 1`)

| `D == 1` | Existing `_pattern_runs!` / `_fill_runs!` (offset-applying 1-D base). |
| `D > 1`  | New `_pattern_runs_intersect!` / `_fill_runs_intersect!`.            |

The new intersection variants are simpler: two-pointer sweep on the
dim-1 interval vectors, emitting at every overlap with the
`coef_selected` value (fill) or just a row-compact index (pattern).
They do not loop over offsets — by the time we reach the base case
with `D > 1`, the offset choice was already made at dim `D`.

Symmetric naming for clarity:
- `_pattern_runs!`, `_fill_runs!` — offset-applying (existing).
- `_pattern_runs_intersect!`, `_fill_runs_intersect!` — zero-shift (new).

#### Coef threading (fill only)

When `D > 1`, the offset `k` that selects a row at dim `D` is fixed by
the outer-dim recursion. The inner recursion (dims `< D`) needs to
write `coefs[k]` at the base. The `_fill_fused!` signature carries an
extra `coef_selected::T` parameter, threaded down at `Nd ≠ D` levels
and set at `Nd == D`.

For `D == 1`, no threading needed — the existing `_fill_runs!` knows
which `k` is in flight because it loops over offsets directly at the
base.

#### Type stability

Both `Nd` (from the interval-tuple length) and `D` (from `st`'s
`LinearStencil{D}` type parameter) are type-level integers. The
`Nd vs D` branches in `_pattern_fused!` / `_fill_fused!` constant-fold
during specialization; only the relevant branch generates code. Verify
with `@code_warntype` after implementation.

#### Implementation steps

1. Write `_pattern_runs_intersect!` (1-D base, two-pointer intersection,
   no offsets).
2. Write `_fill_runs_intersect!` (same, writes `coef_selected` at each
   overlap cell).
3. Write `_pattern_fused!` (recursive N-D). Branches on `Nd vs D`; base
   case dispatches to `_pattern_runs!` (D = 1) or
   `_pattern_runs_intersect!` (D > 1).
4. Write `_fill_fused!` (recursive N-D with `coef_selected` threading).
5. Add N-D methods that delegate to the fused kernels. They mirror the
   existing 1-D methods, with `CartesianRunIndices{N}` (any `N`) in
   place of `CartesianRunIndices{1}` and the `D ≤ N` check replacing
   the `D == 1` check:

   ```julia
   function assemble(
       st::LinearStencil{D,K,T},
       row::CartesianRunIndices{N},
       col::CartesianRunIndices{N},
       ::Type{SparseMatrixCSC{T,Int}} = SparseMatrixCSC{T,Int},
   ) where {D,K,T,N}
       D <= N || throw(ArgumentError(
           "stencil dimension D=$D exceeds cri dimension N=$N"))
       # build colptr/rowval via _pattern_fused!, allocate nzval
   end

   function update!(
       mat::SparseMatrixCSC{T,Int},
       st::LinearStencil{D,K,T},
       row::CartesianRunIndices{N},
       col::CartesianRunIndices{N},
   ) where {D,K,T,N}
       D <= N || throw(ArgumentError(
           "stencil dimension D=$D exceeds cri dimension N=$N"))
       # fill mat.nzval via _fill_fused!
   end
   ```

   The existing `CartesianRunIndices{1}` methods stay as-is (more
   specific; win dispatch for `N == 1`) — or are subsumed if the N-D
   methods are written to also handle `N == 1` correctly, in which case
   delete the 1-D methods. Decide during implementation; subsuming is
   cleaner if the fused kernel's base case already covers `N == 1`.
6. Tests (2-D and 3-D) for each `D ∈ 1..N`. Compare against
   `stencil_reference` and the generalised oracle (next bullet).
7. Generalise `test/oracle.jl`'s `stencil_naive_x` (currently 1-D, uses
   `spdiagm`) to an N-D `stencil_naive` that builds the full dense
   matrix via direct (row, col) mesh enumeration, then sparsifies.

#### Decisions made

- **`_pattern_runs_intersect!` is a standalone function**, not
  `_pattern_runs!` invoked with `offsets = (0,)`. The reuse is tempting
  but pays one extra loop iteration per cell and obscures intent;
  standalone is clearer and predictably faster.
- **Pass `st` (the `LinearStencil`), not a bare `Val{D}`.** The stencil
  already carries `D` as a type parameter plus the offsets and coefs the
  kernels need, so a single `st` argument replaces three (`Val{D}`,
  `offsets`, `coefs`). The recursion still branches on the type-level
  `D` and the tuple length `Nd`.
- **N-D tests stay in `test/test_stencil.jl`** (append 2-D and 3-D
  testsets) until the file crosses ~250 lines, at which point split into
  `test/test_stencil_1d.jl` + `test/test_stencil_nd.jl`.

#### File-level changes (anticipated)

| File | Action |
|------|--------|
| `src/stencil.jl` | Add `_pattern_fused!`, `_fill_fused!`, `_pattern_runs_intersect!`, `_fill_runs_intersect!`, and N-D method dispatch for `assemble` / `update!`. |
| `test/test_stencil.jl` | Add 2-D and 3-D testsets per the implementation steps. |
| `test/oracle.jl` | Add `stencil_naive` (N-D); keep `stencil_naive_x` (1-D) for the existing checks. |
| `AGENTS.md` | Update Scope; flip "N-D deferred" → "N-D implemented". |
| `README.md` | Update Status. |
| `docs/plan.md` | Status section moves the N-D bullet from Roadmap to Implemented. |

### Further milestones (sketched)

#### Composition / Laplacian

The discrete Laplacian on a Cartesian grid is `Δ = Σ_d ∂²/∂x_d²`. Two
realisation paths:

1. **Sum-of-matrices.** Build `LinearStencil{d}((-1,))` × backward-diff
   chained with `LinearStencil{d}((+1,))` (or similar) per dim, then
   sum the resulting `SparseMatrixCSC`s. Cheap to implement on top of
   the N-D milestone; downside is repeated assembly traversals and
   transient intermediate matrices.
2. **Composite stencil type.** A new `CompositeStencil{N,...}` that
   represents a sum of `LinearStencil{d}`s for various `d`. `assemble`
   walks all stencils together, producing a single sparse matrix with
   the merged sparsity pattern. More code (per-column merge of offset
   sets) but a single matrix at the end.

Decision: defer; pick when the consumer demands a particular shape.

#### Other matrix targets

`assemble`'s trailing `::Type{MT}` parameter is positional for future
extension. Likely candidates:

- `BandedMatrices.BandedMatrix` — fast direct solves for stencils with
  small width on a regular mesh; useful for 1-D problems.
- `Matrix` (dense) — small problems / debugging.

Implementation: new method dispatched on the target type, sharing the
kernel sweep but emitting into the target format's representation.

#### Variable coefficients

Constant coefficients cover most CFD on uniform grids. For non-uniform
grids or position-dependent diffusion, two approaches:

1. **Per-cell coefficient array.** A `VariableStencil{D,K,T,N}` whose
   `coefs::Array{T,N+1}` carries `(coef per stencil term) × (per
   mesh cell)`.
2. **Callable.** `update!` accepts a `coef_fn(cell::CartesianIndex{N},
   k::Int) -> T` and queries it per emission.

Decision: defer.

## Verification approach

Every milestone is verified through three independent routes:

1. **Unit tests** — direct calls on internal kernels with hand-computed
   `colptr` / `rowval` expectations on small fixed masks.
2. **Integration tests** — `assemble` + `update!` against
   `stencil_reference` (dictionary-based brute-force, dim-agnostic).
3. **Cross-check oracle** — `test/oracle.jl` builds a naive
   reference from `spdiagm` (1-D) or N-D mesh enumeration (after the
   N-D milestone) and compares against `assemble` + `update!` on
   random masks.

All three must agree across forward / backward / central operators on
both full and holed masks.

## Notes for future agents

- The pointer-sweep style (`Base.in` / `getindex` are forbidden in the
  kernels) is the package's performance core. Any new kernel must walk
  `cri.intervals[d]` with pointers and compute compact indices via
  `Interval.shift`. See `AGENTS.md`.
- Strict-descending offset order is intrinsic to CSC sortedness and is
  validated at the `LinearStencil` constructor boundary. Don't relax
  it without also accepting an explicit sort step.
- The package does **not** depend on `domain(cri)`. Two cri's are
  interpreted on a shared integer mesh; coherence is the caller's
  responsibility.
