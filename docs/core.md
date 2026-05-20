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
4. **`StarStencil` keeps `N`.** Its `N` is the axis count = tuple length
   = grid rank, fixed by construction even when the coefficient is
   symbolic.
5. **General `Stencil{S}`** carries an `SShift`-offset collection and one
   combined coefficient (`ArrayOrTermLike{SVector{K}}`, `K` = offset
   count). It is the form `GridAlgebra.differentiate` emits; it is
   **narrowed** (`as_linear` / `as_star`) to an assemblable
   `LinearStencil` / `StarStencil`, not assembled directly.
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

# N kept; per-axis containers may differ (NTuple{N, <:ArrayOrTermLike{E}}).
struct StarStencil{L, N, M, E<:SVector{M}, C<:NTuple{N, <:ArrayOrTermLike{E}}, S<:AccessStyle} <: AbstractStencil{S}
    terms::C
end

# General lingua franca; Offs is a tuple of StaticShift, A the combined coefficient.
struct Stencil{S<:AccessStyle, Offs<:Tuple{Vararg{StaticShift}}, A<:ArrayOrTermLike} <: AbstractStencil{S}
    offsets::Offs
    term::A
end
```

`E<:SVector{L}` is the array-of-structs element type (one `SVector` of
all `L`/`M` per-offset coefficients per column); the scalar eltype is
`eltype(E)`. This preserves the [`AGENTS.md`](../AGENTS.md) array-of-structs
layout while letting `A` be symbolic.

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

## The general `Stencil` and narrowing

`differentiate` (in `GridAlgebra`) emits `Stencil{RowAccess}`. Narrowing
to an assemblable type is a type-level inspection of the `SShift`
offsets:

- All offsets single-axis, same `D`, contiguous in their `O` → `as_linear`
  builds `LinearStencil{D}(S, SUnitRange(O_min, O_max), term)`.
- Symmetric per-axis reach `−L … +L`, one offset family per axis → `as_star`.
- Otherwise → `ArgumentError` (no optimized kernel; a future general CSC
  kernel could lift this).

Narrowing reuses the combined `SVector`-valued coefficient term verbatim
— it is already shaped as the array-of-structs coefficient. The
`RowAccess → ColumnAccess` conversion (per-offset shift) is applied
before narrowing; see [`docs/cas.md`](cas.md).

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
- `_pattern!` / `_fill!` / `_pattern_nd!` / `_fill_nd!` /
  `_pattern_nd_star!` / `_fill_nd_star!` — **bodies unchanged**.
- `assemble` / `update!` / `build` methods — signatures re-parameterised
  (`E`, `eltype(E)`, rebound `N`), constrained to concrete `A`.

Test impact:

- `test/test_stencil.jl`: constructor calls and `st.term` / `st.terms`
  accesses are unchanged in meaning; the `Fill(SVector(...), n)`
  coefficients already match the relaxed signature (a `Fill` is an
  `AbstractArray`). Type-parameter-position assertions (if any) need
  updating for the dropped `N`.
- `test/reference.jl`: oracle is value-level, unaffected.
- Add: a symbolic-coefficient stencil constructs but raises `MethodError`
  on `assemble`; `materialize` lowers it and assembly then succeeds.

Sequencing (per decision: refactor now):

1. **StencilCore scaffold.** New package; move `AccessStyle` +
   `AbstractStencil`; add `AbstractTerm{T}`, `ArrayOrTermLike`,
   `StaticPair`/`StaticShift`. Tag a version.
2. **Move stencil structs** + constructors into StencilCore with relaxed
   coefficient types. `CartesianOperators` depends on StencilCore,
   re-exports the names.
3. **Re-parameterise assembly** in `CartesianOperators` (`E`/`eltype(E)`/`N`).
   Run `Pkg.test()` — existing suite green.
4. **Add `Stencil{S}`** + `as_linear` / `as_star` narrowing in StencilCore.
5. **Migrate `AGENTS.md`**: the stencil "Sticky decisions" become
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

**In:** the package split; relaxed coefficient types; the general
`Stencil` + narrowing; assembly re-parameterisation (behaviour
identical); migration of the canonical stencil decisions to StencilCore.

**Out / deferred:** a general CSC kernel that assembles `Stencil{S}`
without narrowing (decision 5 keeps narrowing-only for now); CSR
(`RowAccess`) assembly; `BandedMatrix` / dense targets; stencil
composition.

## Open questions

1. **`Stencil` coefficient shape.** Single combined `SVector`-valued
   term (mirrors Linear/Star), or a per-offset coefficient vector that
   the narrowing step gathers? Lean: combined term, consistent with
   [`docs/cas.md`](cas.md) decision 11.
2. **`update!` vs `fill!`.** The current in-place op is `update!`. Adopt
   `Base.fill!` overloading instead, or keep `update!`? (Surfaced in the
   modification-2 discussion; assumed `update!` pending confirmation.)
3. **Re-export breadth.** Should `CartesianOperators` re-export the full
   StencilCore surface (incl. `Stencil`, `AbstractTerm`, `SShift`) for
   source compatibility, or only the assembly-relevant names?
