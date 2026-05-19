# Interface layer for stencil operators.
#
# This file declares the generic verbs (`assemble`, `update!`, `build`)
# shared by every stencil type. Individual stencil types add their own
# methods to these generics from `linear.jl` / `star.jl`. The only
# concrete method defined here is `build`, which is a one-line wrapper
# around `assemble` + `update!` and therefore type-agnostic.

"""
    assemble(st, row, col) -> SparseMatrixCSC

Build the sparsity pattern (`colptr`, `rowval`) of the operator induced
by `st` between `row` and `col` and allocate `nzval` **uninitialised**.
Call [`update!`](@ref) to populate `nzval`, or use [`build`](@ref) to do
both in one shot.

`row` and `col` are interpreted on a shared integer mesh:
`row[d]`, `col[d] :: AbstractUnitRange{Int}` cover the d-th coordinate
of the operator's row / column index space.

Each stencil type adds its own methods — see [`LinearStencil`](@ref)
and [`StarStencil`](@ref).
"""
function assemble end

"""
    update!(mat, st, row, col) -> mat

Write `mat.nzval` in place by re-walking `row` / `col` with `st`. `mat`
must have been produced by a matching [`assemble`](@ref) (same `st`,
`row`, `col`) so its `colptr` / `rowval` align with the kernel's sweep.

Each stencil type adds its own methods — see [`LinearStencil`](@ref)
and [`StarStencil`](@ref).
"""
function update! end

"""
    build(st, row, col) -> SparseMatrixCSC

Equivalent to `update!(assemble(st, row, col), st, row, col)`. Use when
you don't need the pattern / values split — the returned matrix is
fully populated and ready to use.
"""
build(st, row, col) = update!(assemble(st, row, col), st, row, col)
