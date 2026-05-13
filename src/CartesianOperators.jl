module CartesianOperators

using CartesianRuns
using CartesianRuns: Interval, shift
using SparseArrays

include("x_diff.jl")

export forward_x_pattern, forward_x_fill!

end # module CartesianOperators
