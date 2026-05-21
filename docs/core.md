# Plan: `StencilCore.jl` — shared stencil + term-like vocabulary

Forward-looking design plan for extracting the **stencil type vocabulary**
out of `CartesianOperators` into a small, dependency-light package
(`StencilCore`) shared by both the CSC assembler (`CartesianOperators`)
and the symbolic CAS (`GridAlgebra`, see [`docs/cas.md`](cas.md)).

The move is motivated by one observation: once a symbolic term carries
its materialized element type (`AbstractTerm{T}`, [`docs/cas.md`](cas.md)
modification 1), a stencil's coefficient is interchangeably *an array*
or *a term* with that element type. `SymbolicStencil` then stops being a
parallel mirror of `LinearStencil` and becomes the **same type family**
with a `Term`-valued coefficient. Unifying them requires a common home
for `AbstractStencil`, `AbstractTerm`, and the `ArrayOrTermLike` union.

Companion: [`docs/cas.md`](cas.md) (GridAlgebra — the CAS layer),
[`AGENTS.md`](../AGENTS.md) (current canonical stencil invariants, to be
relocated), [`docs/term.md`](term.md) / [`docs/star.md`](star.md).

## Package topology

```
StencilCore                          (deps: StaticArrays)
  ├── AccessStyle, ColumnAccess, RowAccess
  ├── AbstractStencil{S<:AccessStyle}
  ├── AbstractTerm{T}                 (abstract; "dimension-/size-less array-like, eltype T")
  ├── ArrayOrTermLike{T} = Union{AbstractArray{T}, AbstractTerm{T}}
  ├── StaticPair{D,O}, StaticShift     (type-level offsets; see docs/cas.md mod 3)
  ├── LinearStencil, StarStencil       (relaxed coefficient type; see below)
  └── Stencil{S}                       (general offset-list stencil; lingua franca)
        │
        ├──────────────► CartesianOperators   (deps: StencilCore, SparseArrays, StaticArrays)
        │                  build / assemble / update! — CSC kernels, concrete coeffs only
        │
        └──────────────► GridAlgebra           (deps: StencilCore, AbstractTrees, StaticArrays,
                           Slot/Const/Term/Shifted, simplify,    RuntimeGeneratedFunctions)
                           differentiate, materialize, codegen
```

`StencilCore` is the **root** of the DAG: it owns the abstract `AbstractTerm`
and `ArrayOrTermLike` so neither leaf package depends on the other. The
alternative (stencils staying in `CartesianOperators`, `GridAlgebra`
depending on it) inverts the layering — the CAS would pull in
`SparseArrays` + assembly kernels merely to name a coefficient type.

## Sticky decisions

1. **`AbstractTerm{T}` is abstract, lives in StencilCore.** Concrete
   subtypes (`Slot`, `Const`, `Term`, `Shifted`) live in `GridAlgebra`.
   `T` is the materialized element type (concrete or abstract). See
   [`docs/cas.md`](cas.md) modification 1.
2. **`ArrayOrTermLike{T} = Union{AbstractArray{T}, AbstractTerm{T}}`** is
   the coefficient type of every stencil. A stencil is *assemblable*
   when its coefficient is a concrete `AbstractArray`; *symbolic* when it
   is an `AbstractTerm`.
3. **`LinearStencil` drops `N`.** Grid rank is genuinely unknown for a
   symbolic coefficient and recoverable from a concrete one. The
   coefficient element type is `E<:SVector{L}`; the scalar is
   `eltype(E)`.
4. **`StarStencil` is interlaced with an explicit diagonal.** A single
   coefficient `term::A` (one `SVector{M}` per cell) holds the *whole*
   star: `M = 2NL + 1` entries in reverse-lexicographic offset order with
   the diagonal as one explicit middle slot. `N` is kept as a type
   parameter (`StarStencil{L, N, M, …}`, constructor checks `M = 2NL + 1`).
   This replaces the old per-axis-tuple format whose diagonal was the
   *sum* of per-axis centers — which cannot represent a free diagonal
   term (Helmholtz `k²f`, parabolic `∂ₜ`). See
   [Interlaced StarStencil](#interlaced-starstencil).
5. **General `Stencil{S}`** carries a reverse-lex-ordered `NTuple{M, SShift}`
   of offsets and a matching `SVector{M}` coefficient — the *same layout*
   as the interlaced `StarStencil`. It is the form
   `GridAlgebra.differentiate` emits; it is **narrowed** (`as_linear` /
   `as_star` / future `as_planar`) to an assemblable stencil by a
   shift-pattern match + verbatim `term` copy, not assembled directly.
6. **Assembly dispatches on concrete coefficients.** `build` / `assemble`
   / `update!` (in `CartesianOperators`) constrain the coefficient to
   `AbstractArray`; a symbolic coefficient simply has no method →
   `MethodError` until materialized. Plus the existing `S = ColumnAccess`
   constraint.
7. **`materialize(st)` lowers a symbolic stencil to a concrete one** by
   replacing each `AbstractTerm` coefficient with its materialized
   `LazyArray`. The stencil *type family* is unchanged; only the
   coefficient parameter `A` moves from term to array.

## Relaxed stencil types

```julia
const ArrayOrTermLike{T} = Union{AbstractArray{T}, AbstractTerm{T}}

# N dropped; recovered at the assembly call by unifying A's ndims with row::NTuple{N}.
struct LinearStencil{D, O, L, E<:SVector{L}, A<:ArrayOrTermLike{E}, S<:AccessStyle} <: AbstractStencil{S}
    offsets::SUnitRange{O, L}
    term::A
end

# Interlaced: one SVector{M} per cell holds the whole star (M = 2NL+1) in
# reverse-lex offset order, diagonal as the explicit middle slot. N kept.
struct StarStencil{L, N, M, E<:SVector{M}, A<:ArrayOrTermLike{E}, S<:AccessStyle} <: AbstractStencil{S}
    term::A   # constructor checks M == 2NL + 1
end

# General lingua franca: shifts is a reverse-lex-ordered NTuple{M,SShift};
# term is the matching SVector{M} coefficient (same layout as StarStencil).
struct Stencil{M, C<:NTuple{M, StaticShift}, E<:SVector{M}, A<:ArrayOrTermLike{E}, S<:AccessStyle} <: AbstractStencil{S}
    shifts::C
    term::A
end
```

`E<:SVector{L}` (resp. `SVector{M}`) is the array-of-structs element type
— one `SVector` of all per-offset coefficients per cell; the scalar eltype
is `eltype(E)`. This preserves the [`AGENTS.md`](../AGENTS.md)
array-of-structs layout while letting `A` be symbolic. The single-`term`
`StarStencil` (vs the old per-axis `terms::NTuple{N,…}`) is what makes the
diagonal a first-class coefficient.

## Interlaced StarStencil

The `SVector{M}` per cell is ordered **reverse-lexicographically** by the
offset vector (axis `N` most significant, matching `CartesianIndex`
ordering): the negative-shift (lower-triangular) entries from
furthest-to-closest, the diagonal in the middle slot `(M+1)/2`, then the
positive-shift entries closest-to-furthest. For `L = 2`, `N = 3`
(`M = 13`), the slot ↦ offset map is:

```
slot   1      2      3      4     5    6   7   8   9   10    11    12    13
SShift 3ê₃⁻²  ê₃⁻¹   2ê₂⁻²  ê₂⁻¹  2ê₁⁻ ê₁⁻ 𝟎  ê₁  2ê₁  ê₂   2ê₂   ê₃   2ê₃
       (-2,3) (-1,3) (-2,2) (-1,2)(-2,1)(-1,1)() (1,1)(2,1)(1,2)(2,2)(1,3)(2,3)   # (offset, axis)
```

The diagonal is the single slot 7 — set it freely (Helmholtz `k²`,
parabolic mass term), independent of the off-diagonal coefficients.

**Sort-free assembly.** The row linear index for slot `(o, d)` is
`r = c − o·s_d` (`s_d = ∏_{e<d} n_e`). Under the per-axis guard
`2L ≤ length(row[d])` (⟹ `L < n_d`), the sequence `o·s_d` is *strictly
increasing* in slot index, so `r` is strictly decreasing — the kernel
emits slots in reverse for CSC-ascending rows, with boundary trimming
dropping off-mesh slots. No sort. For `N = 1` the layout is exactly
`LinearStencil`'s ascending-offset `SVector`, so `as_linear` is a verbatim
copy.

## Assembly (stays in `CartesianOperators`)

Method signatures change only in their parameter list — the scalar `T`
is recovered as `eltype(E)`, `N` is rebound from `A` and `row`/`col`;
**kernel bodies are unchanged** (they still read `term[c]::SVector` and
slot `k`).

```julia
# 1-D — N pinned to 1 by A and the NTuple{1}.
function assemble(
    st::LinearStencil{1, O, L, E, A, ColumnAccess},
    row::NTuple{1, AbstractUnitRange{Int}},
    col::NTuple{1, AbstractUnitRange{Int}},
) where {O, L, Ts, E<:SVector{L, Ts}, A<:AbstractArray{E, 1}}
    ...
    nzval = Vector{Ts}(undef, length(rowval))
    ...
end

# N-D — N bound by unifying A<:AbstractArray{E,N} with row::NTuple{N}.
function assemble(
    st::LinearStencil{D, O, L, E, A, ColumnAccess},
    row::NTuple{N, AbstractUnitRange{Int}},
    col::NTuple{N, AbstractUnitRange{Int}},
) where {D, O, L, N, Ts, E<:SVector{L, Ts}, A<:AbstractArray{E, N}}
    ...
end
```

A symbolic coefficient (`A<:AbstractTerm{E}`) matches none of these →
`MethodError`, exactly the desired "materialize first" signal.

The `StarStencil` kernels (`_pattern_nd_star!` / `_fill_nd_star!`) are
**rewritten** for the interlaced layout: per output column, walk the `M`
canonical offsets in reverse (CSC-ascending row), skip off-mesh slots,
emit `term[c][k]` (or the diagonal slot directly). This drops the old
3-way merged-diagonal branching — simpler, and it reads one `SVector{M}`
per column instead of `N` per-axis vectors. `_as_linear` for `N = 1`
copies `term` straight through (layouts coincide).

## The general `Stencil` and narrowing

`differentiate` (in `GridAlgebra`) emits `Stencil{RowAccess}`. Narrowing
to an assemblable type is a type-level inspection of the `SShift`
offsets:

- All offsets single-axis, same `D`, contiguous → `as_linear` builds
  `LinearStencil{D}(S, SUnitRange(O_min, O_max), term)`.
- The canonical star pattern (every axis present with symmetric reach
  `−L … +L`, plus the diagonal) → `as_star`.
- Otherwise → `ArgumentError` (no optimized kernel; a future general CSC
  kernel could lift this).

Because `Stencil`, `StarStencil`, and `LinearStencil` all use the **same**
reverse-lex `SVector{M}` layout, narrowing is a shift-pattern match plus a
**verbatim `term` copy** — no reindexing. The `RowAccess → ColumnAccess`
conversion (per-offset shift) is applied *before* narrowing; see
[`docs/cas.md`](cas.md).

## `materialize` on a stencil

```julia
# GridAlgebra (or the bridge extension): Term coefficient → LazyArray coefficient.
materialize(st::LinearStencil{D,O,L,E,A,S}, pairs) where {D,O,L,E,A<:AbstractTerm,S} =
    LinearStencil{D}(S, st.offsets, materialize(st.term, pairs))
```

The result's coefficient is a `LazyArray{E,N}` (a concrete
`AbstractArray`), so the standard `CartesianOperators` assembly methods
now apply.

## `CartesianOperators` refactor plan

Wide-but-shallow. What **moves** to `StencilCore`:

- `src/term.jl` content → StencilCore (`AccessStyle`, `ColumnAccess`,
  `RowAccess`, `AbstractStencil`). Plus the new `AbstractTerm{T}`,
  `ArrayOrTermLike`, `StaticPair`/`StaticShift`, `Stencil`.
- `LinearStencil` / `StarStencil` **struct definitions** and their
  constructors (with the relaxed coefficient type, dropped/kept `N`).

What **stays** in `CartesianOperators`:

- `SparseArrays` dependency.
- `_pattern!` / `_fill!` / `_pattern_nd!` / `_fill_nd!` (LinearStencil
  kernels) — **bodies unchanged**.
- `assemble` / `update!` / `build` methods — signatures re-parameterised
  (`E`, `eltype(E)`, rebound `N`), constrained to concrete `A`.

The `StarStencil` kernels are **rewritten** (not merely re-signatured)
for the interlaced layout (see [Interlaced StarStencil](#interlaced-starstencil)).

Sequencing:

1. ✅ **StencilCore scaffold.** `AccessStyle` + `AbstractStencil`,
   `AbstractTerm{T}`, `ArrayOrTermLike`, `StaticPair`/`StaticShift`.
2. ✅ **Move stencil structs** into StencilCore with relaxed coefficient
   types; `CartesianOperators` re-exports.
3. ✅ **Re-parameterise assembly** (`E`/`eltype(E)`/`N`); suite green.
   *(Steps 2–3 shipped the per-axis `StarStencil`; step 4 supersedes it.)*
4. ⏭ **Interlaced `StarStencil`.** Redefine in StencilCore
   (`{L, N, M, E, A, S}`, single `term`, `M = 2NL + 1`); rewrite
   `_pattern_nd_star!` / `_fill_nd_star!` and the star `assemble` /
   `update!` / `_as_linear` in CartesianOperators. Tests: keep the
   `sum(LinearStencils)` Laplacian oracle (diagonal = the sum) and **add
   a Helmholtz/parabolic test** (free diagonal) the old format couldn't
   represent.
5. ⏭ **Add `Stencil{M, …}`** + `as_linear` / `as_star` narrowing in
   StencilCore (verbatim `term` copy on a shift-pattern match).
6. ⏭ **Migrate `AGENTS.md`**: the stencil "Sticky decisions" become
   StencilCore's canonical record; `CartesianOperators`' `AGENTS.md`
   keeps only the assembly/kernel invariants and points at StencilCore.

## Public surface (StencilCore)

```julia
# Traits / supertypes
AccessStyle, ColumnAccess, RowAccess, AbstractStencil, AbstractTerm, ArrayOrTermLike

# Offsets
StaticPair, StaticShift               # + aliases SPair, SShift  (see docs/cas.md mod 3)

# Stencils
LinearStencil, StarStencil, Stencil
```

`CartesianOperators` re-exports `LinearStencil`, `StarStencil`,
`AbstractStencil`, `AccessStyle`, `ColumnAccess`, `RowAccess`, and adds
`assemble`, `update!`, `build`.

## Scope

**In:** the package split; relaxed coefficient types; the **interlaced
`StarStencil`** (explicit diagonal) + its kernel rewrite; the general
`Stencil` + narrowing; assembly re-parameterisation; migration of the
canonical stencil decisions to StencilCore.

**Out / deferred:** a general CSC kernel that assembles `Stencil{S}`
without narrowing (narrowing-only for now); a `PlanarStencil` +
`as_planar`; CSR (`RowAccess`) assembly; `BandedMatrix` / dense targets;
stencil composition.

## Open questions

1. **`update!` vs `fill!`.** The current in-place op is `update!`. Adopt
   `Base.fill!` overloading instead, or keep `update!`? (Assumed `update!`
   pending confirmation.)
2. **Re-export breadth.** Should `CartesianOperators` re-export the full
   StencilCore surface (incl. `Stencil`, `AbstractTerm`, `SShift`) for
   source compatibility, or only the assembly-relevant names?
3. **`StarStencil` Laplacian convenience.** Decided: raw `SVector{M}`
   only for now (no per-axis + diagonal helper). Revisit if the raw form
   proves error-prone in practice.
