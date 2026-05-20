# Access-style trait + abstract stencil supertype.
#
# `AccessStyle` discriminates how a stencil's coefficient arrays are
# anchored (column vs row). It lives as a *type parameter on the
# stencil* (the trailing slot, by convention); assemblers dispatch on
# it to enforce that the stencil's anchoring matches the target sparse
# format. The trait is never consulted at element-access time —
# coefficient reads are plain `getindex` on plain `AbstractArray`s.

"""
    AccessStyle

Holy trait reporting how a stencil's coefficient arrays are anchored:

- [`ColumnAccess`](@ref): `terms[k][c_idx...]` is the value at
  **column** mesh position `c_idx`. Required for assembly into
  `SparseMatrixCSC`.
- [`RowAccess`](@ref): `terms[k][r_idx...]` is the value at **row**
  mesh position `r_idx`. Required for assembly into a row-major
  format (CSR; not yet implemented).

The trait is a **type parameter on the stencil**, not a runtime field.
Assemblers dispatch on it; mismatching the access style and the
target sparse format is a dispatch-time error (`MethodError`).
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
