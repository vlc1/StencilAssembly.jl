module CartesianOperators

using SparseArrays
using StaticArrays: SUnitRange

include("stencil.jl")   # interface: declarations + generic build
include("linear.jl")    # LinearStencil
include("star.jl")      # StarStencil (depends on LinearStencil via _as_linear)

export LinearStencil, StarStencil, assemble, update!, build

end # module CartesianOperators
