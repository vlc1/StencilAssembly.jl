module CartesianOperators

using SparseArrays
using StaticArrays: SUnitRange, SVector

include("term.jl")      # AccessStyle trait + AbstractStencil supertype
include("stencil.jl")   # interface: declarations + generic build
include("linear.jl")    # LinearStencil
include("star.jl")      # StarStencil (depends on LinearStencil via _as_linear)

export AccessStyle,
       ColumnAccess,
       RowAccess,
       AbstractStencil,
       LinearStencil,
       StarStencil,
       assemble,
       update!,
       build

end # module CartesianOperators
