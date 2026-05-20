# Plan: `GridAlgebra.jl` — symbolic CAS for grid expressions

Forward-looking design plan for the **CAS layer** of a three-package
split. `GridAlgebra` supplies the concrete symbolic term types and the
operations on them (simplify, differentiate, materialize); it depends on
[`StencilCore`](core.md) for the `AbstractTerm{T}` supertype, the
`AbstractStencil` family, and the `StaticShift` offset vocabulary, and it
feeds the CSC assembler `CartesianOperators` through them.

Companion docs: [`docs/core.md`](core.md) (StencilCore — shared
vocabulary + the `CartesianOperators` refactor), [`AGENTS.md`](../AGENTS.md)
(array-of-structs coefficient layout `term::AbstractArray{SVector{L,T},N}`),
[`docs/star.md`](star.md) (`StarStencil` shape).

## Context

`GridAlgebra.jl` is a small Computer Algebra System for expressions on
N-D Cartesian meshes. Two motivating use cases:

1. **Substitution + compiled evaluation.** Build a symbolic expression
   over slot fields (`ϕ`, `ψ`, …), substitute concrete `AbstractArray`s,
   and evaluate elementwise — via a generated per-cell kernel, or by
   dumping inspectable Julia source.
2. **Symbolic differentiation onto stencils.** Differentiate a (possibly
   non-local) symbolic expression with respect to a slot, producing a
   `StencilCore.Stencil` whose per-offset partial-derivative coefficients
   lower to `LinearStencil` / `StarStencil`.

The package stays small on purpose: it implements the minimal trinity —
**tree**, **rewriter**, **derivative table** — plus a codegen lowering.
Where a mature ecosystem alternative exists it is named explicitly (see
[Stable alternatives](#stable-alternatives)).

## Sticky decisions

1. **`AbstractTerm{T}` is array-like with element type `T`.** A term is
   "a dimension-/size-less array-like object whose `eltype` is `T`"
   (concrete `T`) or "looks like `T`" (abstract `T`). `T` is the
   **materialized element type**; it is computed at construction (below).
   Grid **rank `N` is not** a type parameter — it is resolved at
   `materialize` from the substituted arrays' `ndims`. `AbstractTerm{T}`
   itself lives in [`StencilCore`](core.md); the concrete subtypes live
   here.
2. **`T` is computed at construction via `promote_op`; `Union{}` throws.**
   `Term(+, (Slot{:a,Number}(), Slot{:b,Number}()))` runs
   `promote_op(+, Number, Number) = Number` →
   `Term{F,A,Number} <: AbstractTerm{Number}`. `Term(SVector, (a, b))` →
   `SVector{2,Number}`. A `promote_op` of `Union{}` (e.g. genuine
   `SVector` component inhomogeneity) raises an `ArgumentError` **at the
   build site**. `eltype(::AbstractTerm{T}) = T` is then trivial.
3. **Constants are wrapped.** `Const{T} <: AbstractTerm{T}`, value as a
   runtime field. Scalars auto-wrap via `convert(AbstractTerm, x) = Const(x)`.
4. **Shifts are type-level (`StaticShift`).** Offsets enter only through
   the DSL functors (`FwdDiff{D}`, …) whose `D` is a type parameter — so
   they are compile-time known, on the same footing as `LinearStencil`'s
   `O`/`L`. `Shifted` carries a `StaticShift` (a normalized tuple of
   `StaticPair{D,O}`); composition, sorting, dedup, zero-drop, and the
   `+`/`-`/`*Int` algebra happen at the type level. `StaticPair`/`StaticShift`
   live in [`StencilCore`](core.md). "`Shifted` above a `Slot`" is a
   normal-form predicate enforced by `simplify`.
5. **StaticArrays is a hard dependency.** `SVector` slots and `SVector(...)`
   **interception** (`SVector(ϕ, ψ) → Term(SVector, (ϕ, ψ))`,
   eltype `SVector{2, promote}`) are first-class — this is what lets a
   differentiated stencil's combined coefficient term carry eltype
   `SVector{L,scalar}` and slot straight into the array-of-structs layout.
6. **Tree interface via [AbstractTrees.jl](https://github.com/JuliaCollections/AbstractTrees.jl).**
7. **Hand-rolled rule rewriter, post-walk to fixed point.** Rule =
   `(::AbstractTerm) -> Union{Nothing, AbstractTerm}`. No pattern AST, no
   e-graphs (upgrade path under [Stable alternatives](#stable-alternatives)).
8. **Symbolic differentiation via a function-keyed table** in the
   ChainRules `frule`-shape convention; no runtime `ChainRulesCore` dep.
9. **Differentiation is row-anchored.** The Jacobian coefficient at
   offset `δ` is `∂F/∂(shifted arg)` evaluated at the **row** index — no
   shifts injected. `differentiate` emits `Stencil{RowAccess}`.
10. **Output is a `StencilCore.Stencil{S<:AccessStyle}`** (the general
    offset-list stencil), not a bespoke `SymbolicStencil`. The bridge
    converts `RowAccess → ColumnAccess` by the exact per-offset shift
    `term_col(δ) = Shifted((D ⇒ −δ), g_δ)`, then **narrows** to
    `LinearStencil` / `StarStencil` for CSC assembly.
11. **Coefficient is one combined `SVector`-valued term.** Over offsets
    `O … O+L−1`, the single term `SVector(c_O, …, c_{O+L−1})` (with
    `Const(0)` padding). By decision 1 its eltype is `SVector{L,scalar}`,
    so it is an `ArrayOrTermLike{SVector{L,scalar}}` and *is* the
    coefficient field of a (symbolic) `LinearStencil`.
12. **`materialize` lowers to compiled code (codegen).** Build a Julia
    `Expr` per term; execute via
    [`RuntimeGeneratedFunctions.jl`](https://github.com/SciML/RuntimeGeneratedFunctions.jl).
    Shifts (read off the `StaticShift` type) become index arithmetic. The
    inspectable string dump is the **same** `Expr` as a function def.
    `LazyArray <: AbstractArray{T,N}` wraps the generated `fn`, the
    substitution `NamedTuple`, and `axes`.
13. **Depends on [`StencilCore`](core.md); the CSC bridge to
    `CartesianOperators` is a package extension** (`GridAlgebraCartesianOperatorsExt`).

## Type hierarchy

`AbstractTerm{T}` is declared in [`StencilCore`](core.md); the concrete
subtypes here:

```julia
struct Slot{S, T}                 <: AbstractTerm{T} end
Slot{S}() where {S} = Slot{S, Number}()                 # T defaults to Number

struct Const{T}                   <: AbstractTerm{T}
    value::T
end

struct Term{F, A<:Tuple{Vararg{AbstractTerm}}, T} <: AbstractTerm{T}
    fn::F
    args::A
end

struct Shifted{Sh<:StaticShift, T, U<:AbstractTerm{T}} <: AbstractTerm{T}
    shift::Sh
    term::U
end
Shifted(::StaticShift{Tuple{}}, term) = term            # zero shift = identity
```

`Term`'s constructor computes `T = promote_op(callable(F), eltype.(args)...)`,
throwing on `Union{}` (decision 2). `Shifted` inherits `T` from its inner
term `U` (a shift never changes element type), so its eltype is
`eltype(U)` by construction.

```julia
# eltype is now just the supertype parameter:
Base.eltype(::Type{<:AbstractTerm{T}}) where {T} = T

# Recover the callable from its (singleton or constructor) type, for promote_op:
_callable(::Type{F})       where {F<:Function} = F.instance
_callable(::Type{Type{X}}) where {X}           = X      # SVector and other constructors
```

Notes:
- `Slot{S,T}` identity is `(S,T)`, but **substitution / `differentiate`
  match on `S` only**; `T` is a contract checked at `materialize`
  (`eltype(pairs[S]) <: T`).
- Abstract `T` (`Real`, `Number`) is first-class and propagates; a
  non-concrete materialized eltype is correct but slow (opt-in
  concreteness check at `materialize`).

## Static shifts (`StaticPair` / `StaticShift`)

Lives in [`StencilCore`](core.md); summarized here as it drives `Shifted`
and the DSL.

```julia
struct StaticPair{D, O} end;   const SPair  = StaticPair      # axis D, offset O
struct StaticShift{P<:Tuple{Vararg{StaticPair}}}              # normalized
    pairs::P                                                  # (redundant with the type)
end;                            const SShift = StaticShift
```

Inner-constructor invariants: pairs sorted by ascending `D`, no duplicate
`D` (same-`D` pairs summed via a recursive type-level fold), no zero `O`
(dropped). Algebra at the type level:

```julia
Base.:+(a::SShift, b::SShift) = SShift(_merge(a.pairs, b.pairs))   # concat + normalize
Base.:-(::SPair{D,O}) where {D,O} = SPair{D,-O}()
Base.:-(a::SShift)            = SShift(map(-, a.pairs))
Base.:-(a::SShift, b::SShift) = a + (-b)
Base.:*(k::Integer, ::SPair{D,O}) where {D,O} = SPair{D, k*O}()
Base.:*(k::Integer, a::SShift)    = SShift(map(p -> k*p, a.pairs))  # k=0 ⇒ all dropped ⇒ SShift{Tuple{}}
```

`_merge` is the type-level sort-merge implementing the invariants (the
`accumulate(::SPair{D,O}, ::Tuple{SPair{D,P},Vararg}) = SPair{D,O+P}()`
fold, generalised). Display sugar (MAX_DIM = 9, display-only):

```julia
const ê₁ = SShift((SPair{1,1}(),)); … ; const ê₉ = SShift((SPair{9,1}(),))
# show(SShift{Tuple{SPair{1,3},SPair{2,1}}}) prints "3ê₁ + ê₂"; same form constructs it.
```

`3ê₁ + ê₂` is `SShift{Tuple{SPair{1,3}, SPair{2,1}}}` — shifts read and
construct like lattice vectors. MAX_DIM bounds only the basis-symbol set,
not `D`.

## Operator overloading

```julia
for op in (:+, :-, :*, :/, :\, :^, :min, :max)
    @eval Base.$op(a::AbstractTerm, b::AbstractTerm) = Term($op, (a, b))
    @eval Base.$op(a::AbstractTerm, b::Number)       = Term($op, (a, Const(b)))
    @eval Base.$op(a::Number,       b::AbstractTerm) = Term($op, (Const(a), b))
end
for op in (:-, :exp, :sin, :cos, :tan, :log, :sqrt, :abs)
    @eval Base.$op(a::AbstractTerm) = Term($op, (a,))
end
StaticArrays.SVector(args::AbstractTerm...) = Term(SVector, args)   # interception
```

Non-local functors build a `StaticShift` (axis-only, dimension-polymorphic):

```julia
struct FwdDiff{D} <: Function end; const δ₊ = FwdDiff
struct BwdDiff{D} <: Function end; const δ₋ = BwdDiff
struct FwdSum{D}  <: Function end; const σ₊ = FwdSum
struct BwdSum{D}  <: Function end; const σ₋ = BwdSum

(::FwdDiff{D})(t::AbstractTerm) where {D} = Term(-, (Shifted(SShift((SPair{D, 1}(),)), t), t))
(::BwdDiff{D})(t::AbstractTerm) where {D} = Term(-, (t, Shifted(SShift((SPair{D,-1}(),)), t)))
(::FwdSum{D})(t::AbstractTerm)  where {D} = Term(+, (Shifted(SShift((SPair{D, 1}(),)), t), t))
(::BwdSum{D})(t::AbstractTerm)  where {D} = Term(+, (t, Shifted(SShift((SPair{D,-1}(),)), t)))

(::FwdDiff)(x::Number) = zero(x);  (::BwdDiff)(x::Number) = zero(x)
(::FwdSum)(x::Number)  = 2x;       (::BwdSum)(x::Number)  = 2x
```

`simplify` restores normal form (shifts pushed to leaves, nested
`Shifted` merged via the `StaticShift` `+`).

## Tree interface

```julia
using AbstractTrees
AbstractTrees.nodevalue(t::Term)    = t.fn;    AbstractTrees.children(t::Term)    = t.args
AbstractTrees.nodevalue(t::Shifted) = t.shift; AbstractTrees.children(t::Shifted) = (t.term,)
AbstractTrees.nodevalue(s::Slot{S}) where {S} = S; AbstractTrees.children(::Slot)  = ()
AbstractTrees.nodevalue(c::Const)   = c.value; AbstractTrees.children(::Const)    = ()
```

## Rewriting & simplification

Rule = `(::AbstractTerm) -> Union{Nothing, AbstractTerm}`; `simplify`
post-walks to a fixed point (or a step budget). Default rules:

1. **Shift composition** — `Shifted(s₁, Shifted(s₂, t)) → Shifted(s₁ + s₂, t)` (type-level `+`); zero shift ⇒ `t` by the `Shifted` outer ctor.
2. **Shift pushdown** — `Shifted(s, Term(f, args)) → Term(f, map(a -> Shifted(s, a), args))`.
3. **Shift over `Const`** — `Shifted(s, Const(v)) → Const(v)`.
4. **Identity / annihilator** — `+0`, `*1`, `*0`, `--`, …
5. **Constant folding** — `Term(f, (Const(a), Const(b))) → Const(f(a, b))` for allow-listed `f`.

Equality `==` is structural; semantic equivalence is `simplify(a) == simplify(b)`.

## Differentiation

`differentiate(t::AbstractTerm, ::Slot{S}) :: Stencil{RowAccess}` — matches
on the symbol `S` only:

```julia
differentiate(t::AbstractTerm, ::Slot{S}) where {S} = _diff(simplify(t), Val(S))
```

Algorithm (normal-form input ⇒ shifts on leaves):

1. Recurse, collecting `(StaticShift, partial)` contributions:
   - `Const` ⇒ none.
   - `Slot{S₂}` vs `S` ⇒ `(SShift{Tuple{}}, Const(1))` iff `S₂ === S`.
   - `Shifted(sh, Slot{S₂})` vs `S` ⇒ `(sh, Const(1))` iff `S₂ === S`.
   - `Term(f, args)` ⇒ for each `i`, recurse on `argsᵢ`, multiply by
     `derivative(f, Val(i), args...)`, `simplify`.
2. Group by offset, sum partials, drop offsets summing to `Const(0)`.
   **No shift injected** — each `g_δ` is the row-anchored coefficient.
3. Assemble the combined `SVector`-valued coefficient term and wrap in
   `Stencil{RowAccess}` (offsets = the collected `StaticShift`s).

### Why row-anchored (worked)

For `out[i] = ψ[i]·(ϕ[i+1] − ϕ[i])`: `A[i,i+1]=ψ[i]` (δ=+1),
`A[i,i]=−ψ[i]` (δ=0) — coefficients `∂F` at the row `i`, no shifts.
`ColumnAccess` (needed for CSC) injects the offset:
`term_col(δ)[c] = g_δ[c−δ] = Shifted((D ⇒ −δ), g_δ)[c]` — done in the bridge.

### Derivative-rule table (excerpt)

```julia
derivative(::typeof(+), ::Val{i}, xs...) where {i} = Const(1)
derivative(::typeof(-), ::Val{1}, x)               = Const(-1)
derivative(::typeof(-), ::Val{1}, x, y)            = Const(1)
derivative(::typeof(-), ::Val{2}, x, y)            = Const(-1)
derivative(::typeof(*), ::Val{1}, x, y)            = y
derivative(::typeof(*), ::Val{2}, x, y)            = x
derivative(::typeof(/), ::Val{1}, x, y)            = Term(/, (Const(1), y))
derivative(::typeof(^), ::Val{1}, x, n)            = Term(*, (n, Term(^, (x, Term(-, (n, Const(1)))))))
derivative(::typeof(sin), ::Val{1}, x)             = Term(cos, (x,))
derivative(::typeof(cos), ::Val{1}, x)             = Term(-, (Term(sin, (x,)),))
derivative(::typeof(exp), ::Val{1}, x)             = Term(exp, (x,))
derivative(::typeof(log), ::Val{1}, x)             = Term(/, (Const(1), x))
# Users extend: GridAlgebra.derivative(::typeof(my_fn), ::Val{1}, x) = ...
```

## Materialize & `LazyArray`

```julia
struct LazyArray{T, N, F, Args<:NamedTuple} <: AbstractArray{T, N}
    fn::F                                   # RuntimeGeneratedFunction: (args, I::Vararg{Int,N}) -> T
    args::Args
    axes::NTuple{N, UnitRange{Int}}
end
Base.size(la::LazyArray)             = length.(la.axes)
Base.axes(la::LazyArray)             = la.axes
Base.IndexStyle(::Type{<:LazyArray}) = IndexCartesian()
@inline function Base.getindex(la::LazyArray{T,N}, I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(la, I...)
    la.fn(la.args, I...)
end
```

`materialize(term, pairs::NamedTuple)`:

1. **Resolve `N`** from substituted slots' `ndims` (require agreement);
   check `eltype(pairs[S]) <: T` per slot.
2. **Build the `Expr`** over index vars `(i₁,…,i_N)`:
   - `Slot{S}` ⇒ `:(args.$S[$(idxvars...)])`.
   - `Shifted(sh, Slot{S})` ⇒ index arithmetic from `sh`'s `StaticPair{D,O}`
     parameters (read off the type), expanded to `N` (zeros elsewhere).
   - `Const(v)` ⇒ the literal; `Term(f, args)` ⇒ `:( $f($(rendered...)) )`.
3. **Compile** via `@RuntimeGeneratedFunction`.
4. **Axes** = intersection of substituted arrays' axes, shrunk inward by
   the term's shift footprint (no implicit broadcast; empty ⇒ error).
5. **`T`** = `eltype(typeof(term))` (decision 1), cross-checked against
   `Base.infer_return_type(fn, …)`.

### Source dump

`code_string(term; name::Symbol)` renders the **same** `Expr` as a named
function definition (droppable into a `.jl` file):

```julia
function advect((; ϕ, ψ), i)
    ψ[i] * (ϕ[i + 1] - ϕ[i])
end
```

`materialize` and `code_string` share the `Expr`-builder — what runs is
what you read.

### Materialize on a stencil

`materialize(st::LinearStencil{…,A<:AbstractTerm,…}, pairs)` replaces the
`Term` coefficient with its `LazyArray`, returning a concrete-coefficient
`LinearStencil` (now assemblable). See [`docs/core.md`](core.md).

## The bridge (`GridAlgebraCartesianOperatorsExt`)

`build_stencil(sst::Stencil{S}, pairs; access = ColumnAccess)`:

1. If converting `RowAccess → ColumnAccess`: replace each offset `δ`'s
   coefficient `g_δ` with `Shifted((D ⇒ −δ), g_δ)`; re-`simplify`.
2. **Narrow** the `StaticShift` offset set to `LinearStencil{D}` (single
   axis, contiguous) or `StarStencil{L}` (symmetric per-axis reach) via
   `StencilCore`'s `as_linear` / `as_star`; otherwise `ArgumentError`.
3. `materialize` the combined `SVector`-valued coefficient term →
   `LazyArray{SVector{L,T},N}`.
4. Result is an assemblable `LinearStencil`/`StarStencil{…,ColumnAccess}`.

Coefficient axes must cover `col` ([`AGENTS.md`](../AGENTS.md) decision 5):
the `Shifted` conversion shrinks axes by `δ`, so shifted/non-square
operators want an `OffsetArray`-backed slot or mesh-space construction.

## Worked example

```julia
using GridAlgebra, CartesianOperators
using StaticArrays: SVector

ϕ = Slot{:ϕ, Float64}()
ψ = Slot{:ψ, Float64}()

expr  = ψ * δ₊{1}(ϕ)            # out[i] = ψ[i] * (ϕ[i+1] - ϕ[i])
expr′ = simplify(expr)          # Shifted(ê₁, ϕ) on the leaf

print(code_string(expr′; name = :advect))
# function advect((; ϕ, ψ), i)
#     ψ[i] * (ϕ[i + 1] - ϕ[i])
# end

ψv = rand(16); ϕv = rand(16)
lazy = materialize(expr′, (ϕ = ϕv, ψ = ψv))   # LazyArray{Float64, 1}, axes (1:15,)
lazy[1]                                         # ψv[1] * (ϕv[2] - ϕv[1])

sst = differentiate(expr′, ϕ)                   # Stencil{RowAccess}
#   offset ê₁ ↦ ψ ,  offset 0 ↦ -ψ      combined: SVector(-ψ, ψ)

st = materialize(build_stencil(sst, (ψ = ψv,))) # LinearStencil{1,0,2,…,ColumnAccess}
#   column-anchored combined term: SVector(-ψ[c], Shifted((1⇒-1),ψ)[c]) → LazyArray{SVector{2,Float64},1}

A = build(st, (1:15,), (1:15,))
```

## Stable alternatives

| Concern                  | In-house here                                         | Stable alternative                                                                                                          | When to swap                                                                                  |
| ------------------------ | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Symbolic AST             | `AbstractTerm{T}` + `Slot`/`Const`/`Term`/`Shifted`   | [`SymbolicUtils.jl`](https://github.com/JuliaSymbolics/SymbolicUtils.jl) `Sym`, `Term`                                     | If polynomial canonical forms / GCD / factoring become needed                                 |
| Rule rewriter            | Hand-rolled rule fns + postwalk                       | [`SymbolicUtils.jl`](https://github.com/JuliaSymbolics/SymbolicUtils.jl) `@rule` / `Postwalk`                              | When the rule set crosses ~20 rules or pattern variables become unavoidable                   |
| Equivalence              | Normal-form (`simplify(a) == simplify(b)`)            | [`Metatheory.jl`](https://github.com/JuliaSymbolics/Metatheory.jl) e-graphs                                                | When equal expressions have unreachable distinct normal forms                                 |
| Symbolic differentiation | Local `derivative` table (ChainRules shape)           | [`Symbolics.jl`](https://github.com/JuliaSymbolics/Symbolics.jl) `derivative`; [`FastDifferentiation.jl`](https://github.com/brianguenter/FastDifferentiation.jl) | When the registry outgrows the table / higher-order derivatives become routine                |
| AD rule format           | Adopt `frule` *shape*, no dep                         | [`ChainRulesCore.jl`](https://github.com/JuliaDiff/ChainRulesCore.jl)                                                      | When consuming the ecosystem's registered rules                                               |
| Tree traversal           | [`AbstractTrees.jl`](https://github.com/JuliaCollections/AbstractTrees.jl) | (same)                                                                                                                     | n/a                                                                                           |
| Codegen execution        | [`RuntimeGeneratedFunctions.jl`](https://github.com/SciML/RuntimeGeneratedFunctions.jl) | `@generated`; [`GeneralizedGenerated.jl`](https://github.com/JuliaStaging/GeneralizedGenerated.jl); raw `eval` | If RGF's caching / world-age model is a poor fit                                               |
| `Expr`/source rendering  | Custom `code_string` over the shared builder          | `Base.show(::IO, ::Expr)`; Symbolics `build_function`                                                                       | When Symbolics-grade `build_function` (in-place, multi-output) is wanted                       |
| Static offset algebra    | `StaticPair`/`StaticShift` (StencilCore)              | runtime `Tuple{Vararg{Pair}}`; [`StaticArrays.jl`](https://github.com/JuliaArrays/StaticArrays.jl) `SVector` index math    | If type-level shift variety ever stresses compile latency                                     |
| Lazy arrays              | `LazyArray <: AbstractArray` over the generated fn    | [`LazyArrays.jl`](https://github.com/JuliaArrays/LazyArrays.jl); [`OffsetArrays.jl`](https://github.com/JuliaArrays/OffsetArrays.jl) | When broadcast semantics / wider interop is wanted                                            |

## Public surface

```julia
# Concrete term types (subtypes of StencilCore.AbstractTerm)
Slot, Const, Term, Shifted

# Operators (+ Unicode aliases δ₊ δ₋ σ₊ σ₋); basis-shift symbols ê₁ … ê₉
FwdDiff, BwdDiff, FwdSum, BwdSum

# Operations
simplify, differentiate, materialize, code_string

# Customization hook
derivative
```

Re-exported from StencilCore for convenience: `AbstractTerm`, `Stencil`,
`StaticPair`, `StaticShift`, `AccessStyle`/`ColumnAccess`/`RowAccess`.

Tests:
1. Tree construction: operators, `Const` promotion, `Slot{S,T}` eltype,
   `SVector` interception, `promote_op` `Union{}` ⇒ throw at construction.
2. `StaticShift` algebra: `+`/`-`/`*Int`, normalization invariants
   (sort/dedup/zero-drop), `Shifted(zero)=identity`, `show` round-trip
   (`3ê₁ + ê₂`).
3. `simplify` rules (shift composition / pushdown, identity, folding).
4. Differentiation vs hand-computed coefficients (linear, product, chain,
   nonlinear); assert `Stencil{RowAccess}`.
5. `materialize` codegen: axis intersection + shift shrinkage, eltype,
   `getindex`; `code_string` round-trips to a runnable fn.
6. End-to-end: `differentiate → build_stencil (RowAccess→ColumnAccess +
   narrow) → materialize → assemble`, compared to a hand-built oracle.

## Scope

**In:** concrete `AbstractTerm{T}` types; type-level `StaticShift`;
operator overloads + `SVector` interception; `simplify`; `differentiate
→ Stencil{RowAccess}`; codegen `materialize` + `code_string`; the bridge
(RowAccess→ColumnAccess conversion + narrowing) to `LinearStencil` /
`StarStencil`.

**Deferred:** pattern-AST rule language; e-graphs; higher-order
differentiation (algorithmically supported, untested); interpretive
(non-codegen) `materialize`; non-axis-aligned offsets; partial
substitution; `RowAccess` assembly (awaits CSR in `CartesianOperators`);
symbolic stencil composition.

## Open questions

1. **`code_string` API surface.** Standalone `code_string(term; name)`
   vs `Base.show(io, ::MIME"text/x-julia", term)` vs a Symbolics-style
   `build_function`. Lean: `code_string` now.
2. **Operator names.** Export ASCII (`fdiff`, …) with Unicode aliases
   (`δ₊`, …) opt-in, or Unicode by default?
3. **Concreteness policy.** `materialize` *error* / *warn* / *silent* on
   an abstract inferred eltype? (Decision 1 makes the check cheap.)
4. **RGF compile amortization.** Memoize the generated fn keyed on the
   `Expr`? Defer until measured.
5. **`StaticShift` display.** Confirm MAX_DIM = 9 for `ê₁ … ê₉`; what to
   do for `D > 9` (fall back to `SShift{…}` raw show, or `ê₁₀`-style?).
