module CartesianOperators

using CartesianRuns
using CartesianRuns: Interval, shift
using SparseArrays

include("stencil.jl")

export LinearStencil, assemble, update!

end # module CartesianOperators
