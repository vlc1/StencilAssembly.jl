# AGENTS.md

## Project goal

Build a Julia package that assembles `SparseMatrixCSC` operators for stencil
patterns on rectangular Cartesian meshes, where row and column index sets are
`NTuple{N, AbstractUnitRange{Int}}`. Sparsity pattern and numerical fill are
exposed as separate operations; both are allocation-free at the kernel level.

- Package name: **`CartesianOperators.jl`**.
- Forward-looking plan: [`docs/plan.md`](docs/plan.md).
- Design rationale (historical): [`docs/superpowers/specs/2026-05-12-cartesian-operators-design.md`](docs/superpowers/specs/2026-05-12-cartesian-operators-design.md).

## Files

| File                          | Role                                                          |
| ----------------------------- | ------------------------------------------------------------- |
| `src/CartesianOperators.jl`   | Module entry; imports; `include`s; exports                    |
| `src/stencil.jl`              | `LinearStencil`, `assemble`, `update!`, `build`, 1-D kernels  |
| `test/runtests.jl`            | Test suite entry (run via `Pkg.test()`)                       |
| `test/reference.jl`           | Brute-force `stencil_reference` helper                        |
| `test/oracle.jl`              | Standalone cross-check script (not in `Pkg.test()`)           |
| `test/test_stencil.jl`        | LinearStencil + assemble + update! test sets                  |

## Sticky decisions (do not re-litigate)

### Type-driven API

The package's main abstraction is
`LinearStencil{D,K,T,N,C<:NTuple{K,AbstractArray{T,N}}}`:

- `D::Int` — mesh dimension the stencil acts on (1-based).
- `K::Int` — number of stencil terms.
- `T` — shared element type of every coef array.
- `N` — coef-array dimensionality; matches the row/col
  `NTuple{N, AbstractUnitRange{Int}}` at assembly time.
- `offsets::NTuple{K,Int}` — strictly descending 1-D offsets along dim `D`.
- `coefs::C` — `NTuple{K,<:AbstractArray{T,N}}` of coefficient arrays.
  Heterogeneous containers (e.g. `Fill` + `Vector` + `OffsetArray`) are
  fine as long as they share `eltype` and `ndims`.

The inner constructor `LinearStencil{D}(offsets, coefs)` (well-typed path)
validates `D ≥ 1`, `D ≤ N` (stencil dim must fit within coef-array dims),
and strict-descending offsets (`issorted(offsets; lt = >=)`). A catch-all
outer constructor `LinearStencil{D}(::Tuple, ::Tuple)` reports friendly
`ArgumentError`s for ill-typed inputs (length mismatch, non-`Int` offsets,
non-`AbstractArray` coefs, mixed `eltype`, mixed `ndims`). The shared
`eltype` and `ndims` of well-typed coefs are enforced at the method
signature, not in the constructor body.

Three operations:

- `assemble(st, row, col)` — builds `colptr`/`rowval`; allocates `nzval`
  (uninitialised).
- `update!(mat, st, row, col) -> mat` — writes `nzval` in place; allocation-
  free apart from whatever `getindex` on the user's coef arrays costs
  (`Vector`, `Fill`, `OffsetArray` are O(1)).
- `build(st, row, col)` — convenience for `update!(assemble(st, row, col), st, row, col)`;
  returns a fully populated matrix in one shot.

For 1-D, `assemble` / `update!` pin both `D = 1` and `N = 1` at the type
level; misuse turns into `MethodError`. The `D ≤ N` invariant at the
constructor catches the most common misuse (e.g. `LinearStencil{2}` with
1-D coefs) earlier with a friendly `ArgumentError`.

### Row/col representation: rectangular ranges on a shared mesh

`row` and `col` are `NTuple{N, AbstractUnitRange{Int}}` interpreted on a
*single shared* integer mesh — there is no separate "row mesh" or "column
mesh" coordinate system, only rectangular sub-blocks of the same mesh. They
may overlap fully, partially, or not at all; they may be unequal in length
or start at different positions (e.g. `row = (1:5,)`, `col = (3:7,)`).

The matrix size is `prod(length, row) × prod(length, col)`. The compact row
index for mesh position `r` is `r − first(row[d]) + 1` per dimension
(column-major), and analogously for column.

### Coefficient indexing: column anchor on the shared mesh

Each `coefs[k]` is an `AbstractArray{T,N}` indexed by a `CartesianIndex{N}`
mesh position. At each emission, the kernel has both the column's mesh
position `c_idx` and the row's mesh position `r_idx = c_idx − Δ_idx`. By
convention it reads `coefs[k][c_idx]` — the **column anchor**. The choice
is symmetric in cost and matches the CSC sweep order where the column
index is the outer loop.

For non-square operators (e.g., staggered grids, restriction operators)
the coef "naturally" attached to row position `r_idx` for some slots must
be supplied as an array that, when indexed at `c_idx`, yields the value
the user intends for the entry `(r=c−Δ, c)`. The typical recipe is
`OffsetArrays.OffsetArray` to compensate, or to construct the coef array
directly in mesh-space such that `coefs[k][c_idx]` evaluates to the
intended weight. The kernel does not validate axes — accessing outside
the supplied coef array's axes raises `BoundsError` at the emission site.

For constant coefficients use `FillArrays.Fill(value, axes)` — O(1) storage
and O(1) `getindex`.

### Subtraction convention

For column `c` at mesh position `p_c` and stencil offset `Δ`, the matrix
entry lands on the row at mesh position `p_c − Δ`. Stencil offsets are 1-D
`Int` (along dim `D`).

### Boundary policy

Stencil offset → row outside `row` ⇒ that single `(row, col)` entry is
dropped. Columns are sourced from `col` so off-mesh columns are never
visited.

### Kernel shape: segment walk over piecewise-constant per-column nnz

The per-column nnz count

    q(c) = #{k : c_lo_k ≤ c ≤ c_hi_k},
        c_lo_k = max(first(col), rmin + offsets[k])
        c_hi_k = min(last(col), rmax + offsets[k])

is piecewise constant in `c` with at most `2K` breakpoints. Because
offsets are strictly descending, both endpoints are non-increasing in
`k`, so walking `k = k_last, …, k_first` yields two non-decreasing event
streams (lo at `c_lo_k`, hi at `c_hi_k + 1`) — no sort needed. Empty
c-ranges (offsets too large/small to reach any column) form a prefix
and suffix in `k` and are trimmed up front to `[k_first, k_last]` in
O(K).

Within each constant-active segment with start column `prev`, start
pointer `cur`, slope `active = i_hi − i_lo`, and active offsets
`k_a:i_hi` (`k_a = i_lo + 1`), every write is a **closed-form pure
function** of the loop indices — no read-modify-write, no sequential
dependency:

    colptr[c − cmin + 2]                                  = cur + active * (c − prev + 1)
    rowval[cur + active * (c − prev) + (k − k_a)]         = c − offsets[k] − rmin + 1
    nzval [cur + active * (c − prev) + (k − k_a)]         = coefs[k][c]

`cur += active * (e − prev)` between segments. `_pattern!` does the
colptr and rowval writes in one walk; `_fill!` does only the nzval
write and **does not touch `colptr`** — the matrix is consistent
throughout the call. Allocation: `_pattern!` resizes `rowval` once to
the analytic nnz (`Σ (c_hi_k − c_lo_k + 1)` over trimmed `k`); the
kernels are otherwise allocation-free.

CSC sortedness holds because for fixed `c`, slots are ascending in `k`
and the emitted row `c − offsets[k]` is ascending in `k` (offsets
strictly descending) — `SparseMatrixCSC`'s rowval-per-column invariant,
without a sort.

### Offset ordering for CSC sortedness

`SparseMatrixCSC`'s `rowval`-per-column sortedness is achieved without an
explicit sort by requiring stencil offsets to be **strictly descending** at
the `LinearStencil` constructor boundary. Under the subtraction convention,
descending offsets ⇒ ascending rows per column. The check uses
`issorted(offsets; lt = >=)`. The invariant is intrinsic to CSC; CSR would
flip to ascending.

## Conventions

- 1-D kernel name pattern: `_pattern!`, `_fill!`.
- N-D recursive kernel name pattern (deferred): tuple-length dispatch via
  `Base.front` / `last`; accumulators returned as `Tuple`s, no `Ref`s.
- Public API: `LinearStencil`, `assemble`, `update!`, `build`.
- Run tests: `julia --project=. -e 'using Pkg; Pkg.test()'` from package root.

## Scope

Implemented: `LinearStencil{D,K,T,N,C}` constructor (any `1 ≤ D ≤ N`) with
variable coefficients as `AbstractArray{T,N}`; `assemble`, `update!`, and
`build` for 1-D dispatch only (`LinearStencil{1,K,T,1}` against
`NTuple{1, AbstractUnitRange{Int}}` → `SparseMatrixCSC{T,Int}`).

Next milestone (see [`docs/plan.md`](docs/plan.md) for the design):

- **N-D dispatch** — `NTuple{N, AbstractUnitRange{Int}}` × `LinearStencil{D,K,T,N}`
  with `1 ≤ D ≤ N` via recursive dimensional-peeling kernels using
  tuple-length dispatch (`Base.front` / `last`). Branches at each level
  on `Nd vs D`; bottoms out at the 1-D kernel. No `Ref`s — accumulators
  are returned as `Tuple`s.

Further deferred milestones: composition (e.g., Laplacian as a sum of
`LinearStencil`s), non-`SparseMatrixCSC` matrix targets (`BandedMatrix`,
dense).
