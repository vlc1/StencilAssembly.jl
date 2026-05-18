module CartesianOperators

using SparseArrays
using StaticArrays: SUnitRange

include("stencil.jl")

export LinearStencil, assemble, update!, build

end # module CartesianOperators
