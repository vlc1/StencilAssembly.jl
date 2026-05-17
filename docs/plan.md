# CartesianOperators implementation plan

Forward-looking; updated after the move from `CartesianRunIndices` to rectangular
`NTuple{N, AbstractUnitRange{Int}}` row/col representations.
For the historical type-driven API design, see
[`docs/superpowers/specs/2026-05-12-cartesian-operators-design.md`](superpowers/specs/2026-05-12-cartesian-operators-design.md).
The original task-by-task plan at
[`docs/superpowers/plans/2026-05-12-phase0-phase1.md`](superpowers/plans/2026-05-12-phase0-phase1.md)
is historical; superseded.

## Status

Implemented:

- `LinearStencil{D,K,T,N,C<:NTuple{K,AbstractArray{T,N}}}` — variable-
  coefficient stencil. Coefs are `AbstractArray{T,N}` (heterogeneous
  containers allowed: `Fill` + `Vector` + `OffsetArray` etc., as long as
  shared `eltype` and `ndims`). Inner constructor validates `D ≥ 1`,
  `D ≤ N`, and strict-descending offsets via `issorted(offsets; lt = >=)`;
  shared `eltype`/`ndims` enforced at the method signature. A catch-all
  outer constructor reports friendly `ArgumentError`s for ill-typed inputs.
- **Range-based row/col**: row and col are
  `NTuple{N, AbstractUnitRange{Int}}` interpreted on a shared integer
  mesh; matrix size is `prod(length, row) × prod(length, col)`.
- Column-anchor coefficient convention: kernel reads `coefs[k][c_idx]` at
  each emission. `c_idx ∈ CartesianIndices(col)`; user is responsible for
  coef axes covering `col` (wrap with `OffsetArray` when needed).
- `assemble(st, row, col)`, `update!(mat, st, row, col)`, and
  `build(st, row, col)` for `LinearStencil{1,K,T,1}` against
  `NTuple{1, AbstractUnitRange{Int}}`. `assemble`/`update!` pin both
  `D = 1` and `N = 1` at the type level (pure dispatch). `build` is the
  one-shot `update!(assemble(...), ...)` convenience.
- 1-D kernels `_pattern!` and `_fill!` structured as a segment walk
  over the piecewise-constant per-column nnz count
  `q(c) = #{k : c_lo_k ≤ c ≤ c_hi_k}`. Two-pointer merge of lo / hi
  event streams (non-decreasing because offsets are strictly descending);
  empty c-ranges trimmed up front in `O(K)`. Within each constant-active
  segment every `colptr` / `rowval` / `nzval` write is a **closed-form
  pure function** of the loop indices — no read-modify-write, no
  sequential dependency, no mid-call `colptr` corruption. `_fill!` does
  not touch `mat.colptr`. Allocation-free; `_pattern!` resizes `rowval`
  once to the analytic nnz.
- Test suite (39 assertions): constructor validation, `_pattern!`/`_fill!`
  direct kernel tests on equal and shifted ranges, integration tests for
  forward / backward / central x-differences against `stencil_reference`
  (range-based brute-force oracle), Float32 element type, unequal-length
  and shifted-range cases, variable-coefficient density-weighted gradient,
  segment-walk edge cases (`K = 0`, all/partial out-of-range offsets,
  disjoint row/col, coincident events), `D ≤ N` constructor invariant,
  N-mismatch dispatch rejection, and `build` smoke test.

Deliberately not exported / not shipped:

- Named convenience constants (`forward_x`, etc.). Callers construct
  `LinearStencil{1}((1, 0), (Fill(1.0, n), Fill(-1.0, n)))` inline.
- Scalar-coef sugar. The constructor takes `AbstractArray`s only; users
  wrap constants in `FillArrays.Fill`.
- `CartesianRunIndices` / arbitrary-mask support. Row and col are
  rectangular by design; arbitrary masks are not representable.

## Roadmap

### Next milestone — N-D dispatch

Extend `assemble` / `update!` to `LinearStencil{D,K,T,N}` against
`NTuple{N, AbstractUnitRange{Int}}` for any `1 ≤ D ≤ N`. The stencil
acts on mesh dim `D`; the remaining `N − 1` dims are "outer" with zero
shift. The same code path handles every D — `D = 1` is x-aligned,
`D = 2` y-aligned, and so on; a single recursive kernel covers all
`(N, D)` pairs.

The plan is to implement in order, validating each step via tests:

1. **2-D, `D = 1`** — stencil on dim 1, outer dim 2 is intersection.
2. **2-D, `D = 2`** — stencil on dim 2, base dim 1 is intersection.
3. **3-D, `D = 1`** — stencil on dim 1, outer dims 2 and 3 intersection.
4. **3-D, `D = 2`** — stencil at middle dim.
5. **3-D, `D = 3`** — stencil at outermost dim.

#### Kernel structure (recursive, tuple-length dispatch)

Two methods, mirroring `CartesianRuns`'s `_intersect_fused!` shape:

- **Recursive case** dispatches on `NTuple{Nd, AbstractUnitRange{Int}}`
  for any `Nd ≥ 2`. Peels `last(row)` / `last(col)` and recurses with
  `Base.front(row)` / `Base.front(col)`. Internal compile-time branch
  on `Nd == D` chooses between **stencil sweep** (apply per-offset row
  position at this dim) and **intersection sweep** (check `c ∈ row[Nd]`
  with zero shift).
- **Base case** dispatches on `NTuple{1, AbstractUnitRange{Int}}` —
  the innermost dim-1 loop. Internal compile-time branch on `D == 1`
  chooses between **stencil base** (the existing flat 1-D kernel) and
  **intersection base** (per-`k` emission gated on threaded
  per-`k` state).

State threaded through the recursion: scalar accumulators
(`partial_r_lin`, `partial_c_lin`, `active`) plus K-tuples
(`active_per_k`, `r_D_lin_per_k`) that are identity above dim `D`, set
at dim `D`, propagated below. No `Ref`s — accumulators that the inner
recursion needs to advance (e.g. the compact column counter) are
returned as `Tuple` elements.

Both `Nd == D` and `D == 1` are type-level booleans → constant-folded
during specialization; only the relevant branch generates code per
instantiation.

#### File-level changes (anticipated)

| File | Action |
|------|--------|
| `src/stencil.jl` | Add recursive `_pattern!`/`_fill!` methods on `NTuple{Nd, AbstractUnitRange{Int}}` for `Nd ≥ 2`, sharing the existing 1-D base. Add N-D `assemble` / `update!` / `build` methods on `NTuple{N, AbstractUnitRange{Int}}` for any `N`, pinning the `D ≤ N` invariant via dispatch. |
| `test/test_stencil.jl` | Add 2-D and 3-D testsets per the implementation steps. |
| `test/reference.jl` | Already N-D — no change needed. |
| `AGENTS.md` | Flip Scope's "N-D deferred" → "N-D implemented". |
| `docs/plan.md` | Move N-D bullet from Roadmap to Implemented. |

### Further milestones (sketched)

#### Composition / Laplacian

The discrete Laplacian on a Cartesian grid is `Δ = Σ_d ∂²/∂x_d²`. Two
realisation paths:

1. **Sum-of-matrices.** Build per-dim second-difference `LinearStencil`s,
   assemble each into a `SparseMatrixCSC`, and sum. Cheap to implement
   on top of the N-D milestone; downside is repeated assembly traversals
   and transient intermediate matrices.
2. **Composite stencil type.** A new `CompositeStencil{N,...}` representing
   a sum of `LinearStencil{d}`s for various `d`. `assemble` walks all
   stencils together, producing a single sparse matrix with the merged
   sparsity pattern. More code (per-column merge of offset sets) but a
   single matrix at the end.

Decision: defer; pick when the consumer demands a particular shape.

#### Other matrix targets

The trailing `::Type{MT}` parameter that used to live on `assemble` has
been dropped (YAGNI). When a second matrix format actually motivates the
API shape, re-introduce it then. Likely candidates:

- `BandedMatrices.BandedMatrix` — fast direct solves for stencils with
  small width on a regular mesh; useful for 1-D problems.
- `Matrix` (dense) — small problems / debugging.

Implementation: new method dispatched on the target type, sharing the
kernel sweep but emitting into the target format's representation.

#### GPU portability

The range-based kernels and the planned recursive N-D variant are
allocation-free at the kernel level and use only integer arithmetic
plus `getindex` on coef arrays. Both translate naturally to GPU
kernels with appropriate `CartesianIndex` parallelism. Concrete plan
deferred until the N-D milestone is in.

## Verification approach

Every milestone is verified through two independent routes:

1. **Unit tests** — direct calls on internal kernels (`_pattern!`,
   `_fill!`) with hand-computed expectations on small fixed ranges.
2. **Integration tests** — `assemble` + `update!` (or `build`) against
   the brute-force `stencil_reference` (range-based; dim-agnostic).
   Cover equal-length, unequal-length, and shifted-range cases.

Both must agree across forward / backward / central operators on the
covered range shapes.

## Notes for future agents

- Strict-descending offset order is intrinsic to CSC sortedness and is
  validated at the `LinearStencil` constructor boundary. Don't relax
  it without also accepting an explicit sort step. The segment-walk
  event streams (lo at `c_lo_k`, hi at `c_hi_k + 1`) rely on it being
  non-increasing in `k`, and the in-segment slot ordering
  (`k − k_a` ascending) gives the row-ascending CSC invariant.
- The package does **not** depend on `CartesianRuns`. Row and col are
  rectangular subsets on a shared integer mesh; coherence of the
  shared-mesh interpretation is the caller's responsibility.
- For the upcoming N-D recursion, prefer tuple-length dispatch
  (`Base.front` / `last`, separate methods for `NTuple{1, …}` base
  and `NTuple{Nd, …}` recursive) over passing `Val{N}` or runtime
  `Nd` indices around. Accumulators returned as `Tuple`s, no `Ref`s.
  The segment-walk shape generalises naturally — at each level the
  active k-range is a function of the dim-`D` projection only, so the
  segment walk runs once per outer-product position with closed-form
  slot offsets across `c_1, …, c_{D−1}, c_{D+1}, …, c_N`.
