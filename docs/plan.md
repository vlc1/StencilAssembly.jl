# CartesianOperators implementation plan

Forward-looking. Reflects the move from arbitrary `NTuple{K, Int}`
offsets to contiguous `SUnitRange{O, L}` and the rewrite of the 1-D
kernel from event-merge segment walk to the three-phase shape.
Historical: [`docs/superpowers/specs/2026-05-12-cartesian-operators-design.md`](superpowers/specs/2026-05-12-cartesian-operators-design.md),
[`docs/superpowers/plans/2026-05-12-phase0-phase1.md`](superpowers/plans/2026-05-12-phase0-phase1.md)
(superseded).

## Status

**Implemented** (canonical spec in [`AGENTS.md`](../AGENTS.md)):

- `LinearStencil{D, O, L, T, N, C<:NTuple{L, AbstractArray{T, N}}}` with
  contiguous unit-stride offsets (`SUnitRange{O, L}`) and ascending coefs.
- 1-D `assemble` / `update!` / `build` (`LinearStencil{1, O, L, T, 1}` ×
  `NTuple{1, AbstractUnitRange{Int}}` → `SparseMatrixCSC{T, Int}`) with
  the `L − 1 ≤ length(row[1])` runtime guard.
- Three-phase contiguous kernel: closed-form interior `cur(c)`, ramps
  with `max(0, active)` off-mesh clipping. One `resize!`; otherwise
  allocation-free.
- Test suite: constructor, direct kernels, integration vs the
  brute-force oracle, edge cases (`L = 0`, full/partial trim, off-mesh
  tails, the `L = m + 1` boundary, the `L > m + 1` guard), `D ≤ N`
  invariant.

Deliberately out of scope: named convenience constants, scalar-coef
sugar, non-contiguous offsets, arbitrary-mask row/col.

## Roadmap

### Next milestone — N-D dispatch

Extend `assemble` / `update!` to `LinearStencil{D, O, L, T, N}` against
`NTuple{N, AbstractUnitRange{Int}}` for any `1 ≤ D ≤ N`. The stencil
acts on mesh dim `D`; the other `N − 1` dims are "outer" with zero
shift. Implementation order, validated via tests at each step:

1. 2-D, `D = 1` &nbsp; 2. 2-D, `D = 2` &nbsp; 3. 3-D, `D = 1` &nbsp;
4. 3-D, `D = 2` &nbsp; 5. 3-D, `D = 3`.

Kernel structure: a **recursive case** on
`NTuple{Nd, AbstractUnitRange{Int}}` for `Nd ≥ 2` peels
`last(row)`/`last(col)` and recurses via `Base.front`; a compile-time
branch on `Nd == D` chooses **stencil sweep** vs **intersection sweep**.
A **base case** on `NTuple{1, ...}` branches on `D == 1` between the
1-D three-phase kernel and an intersection base. State threaded as
scalar accumulators plus L-tuples (`Tuple`s, no `Ref`s); both
type-level branches constant-fold.

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
