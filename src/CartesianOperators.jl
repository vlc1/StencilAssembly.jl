module CartesianOperators

using SparseArrays
using StaticArrays: SUnitRange, SVector
using StencilCore

include("stencil.jl")   # interface: declarations + generic build
include("linear.jl")    # LinearStencil assembly (type lives in StencilCore)
include("star.jl")      # StarStencil assembly (delegates 1-D via _as_linear)

# Re-export the stencil vocabulary owned by StencilCore.
export AccessStyle,
       ColumnAccess,
       RowAccess,
       AbstractStencil,
       LinearStencil,
       StarStencil,
       Stencil,
       as_linear,
       as_star

# CartesianOperators' own assembly verbs.
export assemble,
       update!,
       build

end # module CartesianOperators
