# Plan: `AbstractStencil{S<:AccessStyle}` + `S` as last type parameter

> **Note (partially superseded).** The `S<:AccessStyle` /
> `AbstractStencil` design below is current. The **coefficient storage**
> shown in the struct signatures (`terms::NTuple{L, AbstractArray{T, N}}`
> for `LinearStencil`, `NTuple{N, NTuple{M, AbstractArray{T, N}}}` for
> `StarStencil`) has since changed from struct-of-arrays to
> array-of-structs: `LinearStencil` now holds `term::A` with
> `A<:AbstractArray{SVector{L, T}, N}`, and `StarStencil` holds
> `terms::NTuple{N, AbstractArray{SVector{M, T}, N}}`. Element access
> `terms[k][c]` became `term[c][k]`. See [`AGENTS.md`](../AGENTS.md) for
> the current canonical spec.

## Context

Two stencil types live in the package (`LinearStencil`, `StarStencil`).
Both represent matrix terms as `(offset, N-D coefficient array)`. The
package currently commits to **column anchoring** at the kernel level
— `coefs[k][c_idx...]` is the value at column mesh position `c_idx`
for offset `δ_k`. A future CSR backend would want **row anchoring**.

Refactor the stencil types so:
1. An abstract supertype `AbstractStencil{S<:AccessStyle}` groups
   them; the `AccessStyle` trait accessor lives once at the
   supertype.
2. Concrete stencils carry `S` as the **last** type parameter and
   subtype `AbstractStencil{S}`.
3. The field name `coefs` becomes `terms` (the offset/array pair *is*
   a term of the stencil's polynomial; the storage of the array
   component is what `terms` holds).
4. Construction uses a **positional `Type` tag** for the access
   style; default is `ColumnAccess`.
5. `assemble` / `update!` (CSC path) constrain to
   `AbstractStencil{ColumnAccess}` via dispatch on each concrete
   type. A `RowAccess` stencil routed to CSC assembly fails with a
   natural `MethodError`.

## Reflection — converging on the minimal design

Five iterations brought us here:

1. `StencilTerm` carries `shift` — redundant with `offsets`.
2. `StencilTerm` without `shift`, kernel passes shift — still threads
   state through the hot path.
3. `StencilTerm <: AbstractArray` with construction-time access-style
   check — kernel reads via plain `getindex`.
4. `S` hoisted from term to stencil — `StencilTerm` dissolves; coefs
   become plain arrays again.
5. **This plan:** add an abstract supertype, settle `S` at the
   stencil's last type-parameter position, default access style via
   positional `Type` tag in the constructor.

Each round moved metadata outward and eliminated redundancy. The
final form: the package looks **exactly as it does today** at the
kernel level, plus one new type parameter `S<:AccessStyle` on each
stencil and a 15-line abstract-supertype file.

## Settled decisions

- **Offset sign:** unchanged. `δ = c − r` (diagonal number; for
  column `j` and row `i`, the diagonal is `k = j − i`). Docstrings
  add this clarifying line.
- **RowAccess on CSC:** natural `MethodError`; no friendly stub.
- **Field name:** `coefs` → `terms`.
- **Abstract supertype:** `AbstractStencil{S<:AccessStyle}` with the
  trait accessor defined once at the supertype level.
- **`S` position:** last in each concrete stencil's type-parameter
  list. Accessed via `AccessStyle(typeof(st))` (position-independent).
- **Constructor surface:** positional `Type` tag for `S`, default
  `ColumnAccess`. `LinearStencil{D}(RowAccess, offsets, terms)` and
  `StarStencil{L}(RowAccess, terms)` are the explicit forms.
- **No `StencilTerm` type** — eliminated.

## Part 1 — Spec

### `src/term.jl`

```julia
"""
    AccessStyle

Holy trait reporting how a stencil's coefficient arrays are anchored:

- [`ColumnAccess`](@ref): `terms[k][c_idx...]` is the value at
  **column** mesh position `c_idx`. Required for assembly into
  `SparseMatrixCSC`.
- [`RowAccess`](@ref): `terms[k][r_idx...]` is the value at **row**
  mesh position `r_idx`. Required for assembly into a row-major
  format (CSR; not yet implemented).

The trait is a **type parameter on the stencil**, not a runtime
field. Assemblers dispatch on it. Mismatching the access style and
the target sparse format is a dispatch-time error
(`MethodError`).
"""
abstract type AccessStyle end

struct ColumnAccess <: AccessStyle end
struct RowAccess    <: AccessStyle end

"""
    AbstractStencil{S<:AccessStyle}

Abstract supertype for every stencil in this package. Subtypes
(`LinearStencil`, `StarStencil`, …) carry the access style `S` as
their **last** type parameter and declare `<: AbstractStencil{S}`.

Provides the [`AccessStyle`](@ref) trait accessor — subtypes inherit
it without redefining.
"""
abstract type AbstractStencil{S<:AccessStyle} end

AccessStyle(st::AbstractStencil) = AccessStyle(typeof(st))
AccessStyle(::Type{<:AbstractStencil{S}}) where {S} = S()
```

### Updated `LinearStencil` (in `src/linear.jl`)

```julia
"""
    LinearStencil{D, O, L, T, N, C<:NTuple{L, AbstractArray{T, N}}, S<:AccessStyle}
        <: AbstractStencil{S}

Variable-coefficient stencil with **contiguous** offsets aligned with
mesh dimension `D`. Offsets are **diagonal indices** in the
numerical-linear-algebra sense: for a column `j` and a row `i` the
diagonal number is `k = j − i`. The entry at column `c`, offset `δ`
lands on row `c − δ`.

Type parameters:
- `D`: mesh dim along which the stencil acts (1-based, `1 ≤ D ≤ N`).
- `O = δ_min`, `L = δ_max − δ_min + 1` — encoded in `SUnitRange`.
- `T`, `N`: shared coef `eltype` / `ndims`.
- `C`: concrete coef-tuple type.
- `S<:AccessStyle`: anchoring of the coefficient arrays
  (`ColumnAccess` for CSC, `RowAccess` for CSR).

# Construction

```julia
# Default (ColumnAccess):
LinearStencil{1}(SUnitRange(0, 1), (Fill(-1.0, 5), Fill(1.0, 5)))

# Explicit access style (positional Type tag):
LinearStencil{1}(RowAccess, SUnitRange(0, 1), (Fill(-1.0, 5), Fill(1.0, 5)))
```
"""
struct LinearStencil{D, O, L, T, N,
                     C<:NTuple{L, AbstractArray{T, N}},
                     S<:AccessStyle} <: AbstractStencil{S}
    offsets::SUnitRange{O, L}
    terms::C

    function LinearStencil{D}(
        ::Type{S},
        offsets::SUnitRange{O, L},
        terms::NTuple{L, AbstractArray{T, N}},
    ) where {D, S<:AccessStyle, O, L, T, N}
        D isa Int && D >= 1 || throw(ArgumentError(
            "stencil dimension D must be a positive Int (got $D)"))
        D <= N || throw(ArgumentError(
            "stencil dimension D=$D exceeds coef-array dimension N=$N"))
        new{D, O, L, T, N, typeof(terms), S}(offsets, terms)
    end
end

# Default outer ctor — ColumnAccess.
LinearStencil{D}(offsets::SUnitRange, terms::Tuple) where {D} =
    LinearStencil{D}(ColumnAccess, offsets, terms)
```

The existing friendly outer ctors (`coefs::Tuple` catch-all,
non-`SUnitRange` offsets fallback) get an `::Type{S}` first argument
threaded through the default-ColumnAccess path.

### Updated `StarStencil` (in `src/star.jl`)

```julia
"""
    StarStencil{L, T, N, M, C<:NTuple{N, NTuple{M, AbstractArray{T, N}}}, S<:AccessStyle}
        <: AbstractStencil{S}

N-D star-shaped stencil with symmetric reach `−L … +L` per axis.
Offsets per axis are `−L:L` interpreted as diagonal indices
(`δ = c − r` along that axis).

# Construction

```julia
StarStencil{1}(coefs_tuple)                      # default ColumnAccess
StarStencil{1}(RowAccess, coefs_tuple)           # explicit
```
"""
struct StarStencil{L, T, N, M,
                   C<:NTuple{N, NTuple{M, AbstractArray{T, N}}},
                   S<:AccessStyle} <: AbstractStencil{S}
    terms::C

    function StarStencil{L}(
        ::Type{S},
        terms::NTuple{N, NTuple{M, AbstractArray{T, N}}},
    ) where {L, S<:AccessStyle, T, N, M}
        L isa Int && L >= 1 || throw(ArgumentError(
            "stencil reach L must be a positive Int (got $L)"))
        M == 2L + 1 || throw(ArgumentError(
            "coefs inner tuple length must be 2L+1=$(2L + 1) (got $M)"))
        new{L, T, N, M, typeof(terms), S}(terms)
    end
end

StarStencil{L}(terms::Tuple) where {L} = StarStencil{L}(ColumnAccess, terms)
```

### Assemble / update! signatures

Each method gains a `ColumnAccess` constraint via the type parameter:

```julia
function assemble(
    st::LinearStencil{1, O, L, T, 1, <:Any, ColumnAccess},
    row, col,
) where {O, L, T}
    # unchanged kernel body — terms are still plain arrays
end

function update!(
    mat::SparseMatrixCSC{T, Int},
    st::LinearStencil{1, O, L, T, 1, <:Any, ColumnAccess},
    row, col,
) where {T, O, L}
    # unchanged
end
```

N-D analog and StarStencil analogs follow the same pattern: the
trailing `S` slot is constrained to `ColumnAccess`; the `C` slot
above it stays `<:Any` (or, equivalently, is dropped to a free
where-var).

`build` is generic; either left unconstrained (current state) or
tightened to `build(st::AbstractStencil, row, col)` for clearer
errors on misuse.

### `_as_linear`

Propagates `S`:

```julia
_as_linear(st::StarStencil{L, T, 1, M, C, S}) where {L, T, M, C, S} =
    LinearStencil{1}(S, SUnitRange(-L, L), st.terms[1])
```

A `StarStencil{L, T, 1, M, C, RowAccess}` becomes a
`LinearStencil{1, …, RowAccess}` — both correctly unassemblable to CSC.

## Part 2 — Soundness check

### Construction syntax mechanics

`LinearStencil{D}(::Type{S}, offsets, terms)` is a constructor
parameterised solely on `D`. `S` enters the dispatch through the
**positional `Type{S}` argument**, not through a `LinearStencil{D, S}`
type-prefix. This sidesteps the position-2-binding pitfall entirely
(no clash with `O`, no need to reorder).

- Default call: `LinearStencil{1}(offsets, terms)` → routes to the
  default-ColumnAccess outer ctor → re-routes to
  `LinearStencil{1}(ColumnAccess, offsets, terms)`.
- Explicit call: `LinearStencil{1}(RowAccess, offsets, terms)` →
  matches the inner ctor directly with `S = RowAccess`.

### Type-inference flow

```
LinearStencil{1}(ColumnAccess, SUnitRange(0, 1), (Fill(-1.0, 5), Fill(1.0, 5)))
  ↓ inner ctor: where {D=1, S=ColumnAccess, O=0, L=2, T=Float64, N=1}
  ↓ new{1, 0, 2, Float64, 1, Tuple{Fill{…}, Fill{…}}, ColumnAccess}(offsets, terms)
  ↓ Instance type: LinearStencil{1, 0, 2, Float64, 1, Tuple{…}, ColumnAccess}
                <: AbstractStencil{ColumnAccess}
```

### Trait dispatch

`AccessStyle(st::LinearStencil{1, 0, 2, …, ColumnAccess})` works via
the supertype's `AccessStyle(::Type{<:AbstractStencil{S}}) where {S}`
— position-independent in the subtype.

### Test-impact audit

- Rename `st.coefs` → `st.terms` in four assertions
  (`test/test_stencil.jl:25–26` and the StarStencil constructor
  block). Mechanical.
- All other tests unchanged in meaning.

### Why no constructor-syntax pitfall

The positional `Type` tag is unambiguous: `LinearStencil{1}(RowAccess,
offsets, terms)` passes `RowAccess` as a normal positional argument
of type `Type{RowAccess}`. The inner ctor's `::Type{S}` slot binds
it. No interaction with type-parameter-position ordering.

## Part 3 — Integration plan

### File layout

- **New:** `src/term.jl` — `AccessStyle`, `ColumnAccess`,
  `RowAccess`, `AbstractStencil`, `AccessStyle(::AbstractStencil)`.
  ~30 lines including docstrings.
- **Modify:** `src/StencilAssembly.jl` — `include("term.jl")`
  before `linear.jl`. Add exports `AccessStyle`, `ColumnAccess`,
  `RowAccess`, `AbstractStencil`.
- **Modify:** `src/linear.jl`:
  - Add `S<:AccessStyle` as the trailing type parameter; declare
    `<: AbstractStencil{S}`.
  - Rename field `coefs` → `terms`.
  - Inner ctor signature: `LinearStencil{D}(::Type{S}, offsets, terms)`.
  - Default outer: `LinearStencil{D}(offsets, terms) =
    LinearStencil{D}(ColumnAccess, offsets, terms)`.
  - Friendly outer ctors thread `S` through via the same default.
  - `assemble` / `update!`: add `ColumnAccess` in the trailing slot.
  - Kernel bodies (`_pattern!`, `_fill!`, `_pattern_nd!`,
    `_fill_nd!`): **unchanged** — `st.terms` is a tuple of plain
    `AbstractArray`s.
  - Docstrings: add the "offsets are diagonal indices: `δ = c − r`"
    clarification.
- **Modify:** `src/star.jl`: analogous structural changes. `_as_linear`
  propagates `S`. Kernels unchanged.
- **Modify:** `test/test_stencil.jl`: rename `coefs` → `terms` in
  the four affected assertions.

### What stays exactly the same

- All kernel bodies.
- The CSC output shape and semantics.
- The offset sign convention (`δ = c − r`).
- The `2L ≤ length(row[d])` guard for `StarStencil`.
- All other tests' meaning.
- Public exports `LinearStencil`, `StarStencil`, `assemble`,
  `update!`, `build` (with additions, not removals).

### Phased rollout

1. **Phase 1 — `src/term.jl`.** Define `AccessStyle`,
   `ColumnAccess`, `RowAccess`, `AbstractStencil`, the trait
   accessor. Include from `StencilAssembly.jl`; export. Module
   loads cleanly.
2. **Phase 2 — `LinearStencil` refactor.**
   - Add `S` last in the struct's type parameters; `<: AbstractStencil{S}`.
   - Rename `coefs` → `terms`.
   - Update inner ctor + default outer ctor + friendly outer ctors.
   - Constrain `assemble` / `update!` to `…, ColumnAccess`.
   - Update docstrings.
   - Rename `st.coefs` → `st.terms` in the affected test assertions.
   - Run `Pkg.test()`; expect all 31 testsets green.
3. **Phase 3 — `StarStencil` refactor.** Same pattern. `_as_linear`
   propagates `S`. Run suite.
4. **Phase 4 — new testsets.**
   - `AccessStyle(st)` returns `ColumnAccess()` for default ctor.
   - `LinearStencil{1}(RowAccess, offsets, terms)` constructs
     without error; `AccessStyle(st) === RowAccess()`.
   - `assemble(::LinearStencil{1, …, RowAccess}, …)` raises
     `MethodError`.
   - Same trio for `StarStencil`.
   - `AbstractStencil` is the common supertype:
     `LinearStencil{1}(…) isa AbstractStencil` is true; ditto
     `StarStencil{1}(…) isa AbstractStencil`.
5. **Phase 5 — docs.**
   - Update `AGENTS.md` "Sticky decisions" with a new item:
     "Access style `S<:AccessStyle` is the trailing stencil
     type-parameter; `AbstractStencil{S}` is the supertype; CSC
     assembly dispatches on `S = ColumnAccess`."
   - Update `docs/plan.md` status.
   - Touch `docs/star.md` if its struct signature is now stale.

Each phase ends with `Pkg.test()` green.

## Part 4 — Verification

1. `julia --project=. -e 'using StencilAssembly'` after each
   phase — no `UndefVarError` / no warnings.
2. `julia --project=. -e 'using Pkg; Pkg.test()'` — full suite green
   after Phases 2, 3, 4. Existing assertions unchanged in meaning;
   ~6 new assertions (Phase 4).
3. Allocation regression: `@allocated update!(...)` for 1-D and 2-D
   `Fill`-coef cases after one warmup — expect 0 bytes (matches
   pre-refactor).
4. Type-stability spot check: `@code_warntype` on
   `update!(mat, ::LinearStencil{1, 0, 2, Float64, 1, <:Any, ColumnAccess}, row, col)`
   — no `Any` / `Union` slots. The kernel is bit-identical to
   pre-refactor in lowered form (the `S` and `C` slots are dead
   weight inside the kernel body).
5. Dispatch sanity:
   - `assemble(LinearStencil{1}(RowAccess, SUnitRange(0, 1),
     (Fill(1.0, 5), Fill(1.0, 5))), (1:5,), (1:5,))` → `MethodError`.
   - `LinearStencil{1}(SUnitRange(0, 1), (Fill(1.0, 5), Fill(1.0,
     5))) isa AbstractStencil` → `true`.

## Non-goals

- Implementing a CSR assembler. The trait enables it; building
  it now is premature.
- Composition / sum / product of stencils. The abstract supertype
  makes future generic operations easier to add but isn't part of
  this refactor.
- Mixed-access-style stencils within one assembler. Single uniform
  `S`.
- A `StencilTerm` wrapper type. Eliminated.
- Sign-convention change for offsets. Settled: keep `δ = c − r`.
- Position-2 placement of `S`. Settled: position last; construction
  via positional `Type` tag.

## Open questions

**None remaining.**
