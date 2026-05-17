module CartesianOperators

using SparseArrays

include("stencil.jl")

export LinearStencil, assemble, update!, build

end # module CartesianOperators
