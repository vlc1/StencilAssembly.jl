# Implementation Notes: N-D Stencil Assembly

## Status (2026-05-18, Updated)

### Completed
- Full 1-D stencil assembly: `LinearStencil{1, O, L, T, 1}` on `NTuple{1, AbstractUnitRange{Int}}`
- Three-phase kernel (`_pattern!`, `_fill!`) with CSC sortedness guarantee
- 31 assertions across 8 testsets, all green
- Constructor validation (`D ≤ N` invariant, guard `L - 1 ≤ length(row[D])`)
- N-D recursive kernel structure: `_pattern_nd_recursive()`, `_fill_nd_recursive()`
- Dispatch on tuple length: Nd-aware overloads in `_pattern_nd!` / `_fill_nd!` / `assemble` / `update!`
- Type-stable recursion without `Ref`s (matching CartesianRuns patterns)

### In Progress / Known Issues
- **`colptr` not properly filled in N-D case**: The current `_pattern_nd_recursive` doesn't update `colptr[j+1]` for every global column j. This is the critical missing piece.
  - **Root cause**: The recursion peels dimensions but doesn't track the global column index as it recurses.
  - **Fix needed**: Thread a mutable `col_idx` counter (or similar) through the recursion, incrementing it for each inner column and writing `colptr[col_idx + 1] = cur` before returning from base case.
  
- **`coefs` indexing in `_fill_nd_recursive`**: Currently just mirrors `_pattern_nd_recursive` structure but doesn't write coefficient values.
  - **Fix needed**: In the base case (Nd=1, D=1), properly index `coefs[k][c1, c2, ...]` using the threaded column mesh coordinates.

### Next Steps

#### Critical: Fix colptr in `_pattern_nd_recursive`
The key is to thread global column indices through the recursion. Pseudocode:
```julia
function _pattern_nd_recursive(..., col_idx_ref::Vector{Int})
    if N_dims == 1
        for c in col[1]  # col is a single range
            _pattern!(rowval, colptr, offsets, row[1], col[1])
            # After _pattern! returns, colptr is filled for all cols in this segment
            # But we need to map them to global column indices
            col_idx_ref[1] += length(col[1])
        end
    else
        for c_last in col_last
            _pattern_nd_recursive(..., col_idx_ref)
        end
    end
end
```

Actually, the problem is deeper: `_pattern!` writes `colptr[c - cmin + 2]` for local indices, not global ones. For N-D, we need to remap these to global indices or rewrite the recursion to build `colptr` incrementally.

#### Option A: Incremental colptr
Instead of calling `_pattern!` directly (which assumes local colptr indexing), manually emit rows and update global `colptr` in the base case.

#### Option B: Post-process colptr
After `_pattern_nd_recursive` completes, convert local indices to global ones. But this loses info about which columns are which.

#### Recommended: Option A
Rewrite the base case to emit rows one-by-one and update `colptr` with global indices.

#### 2-D tests
Once colptr is fixed, add:
```julia
@testset "2-D assemble + update!" begin
    row = (1:3, 1:4); col = (1:3, 1:4)
    for D in 1:2
        st = LinearStencil{D}(SUnitRange(0,1), (Fill(-1.0, 3, 4), Fill(1.0, 3, 4)))
        J = build(st, row, col)
        offsets = ntuple(k -> CartesianIndex(ntuple(d -> d==D ? (k-1) : 0, 2)), 2)
        ref = stencil_reference(offsets, (-1.0, 1.0), row, col)
        @test J == ref
    end
end
```

#### 3-D tests
Same pattern, with 3-D grids and D ∈ {1, 2, 3}.

## Architecture

### Current N-D Recursion
```
assemble(st, row::NTuple{Nd}, col)
  → _pattern_nd!(rowval, colptr, ..., row, col, Val(D), Val(Nd))
    → _pattern_nd_recursive(rowval, colptr, ..., row, col, Val(D))
      [recurse on Nd]
        if Nd == D: three-phase stencil sweep
        else: pure intersection (loop over col_Nd)
      [base case Nd=1, D=1]
        → _pattern!(..., row[1], col[1])  [LOCAL colptr indices!]
```

The bug: `_pattern!` returns with `colptr` indexed 1 to `length(col[1]) + 1`, treating `col[1]` as indices 1 to n. But in N-D we need global column indices `[col_global, col_global + n, ...]`.

### Fix: Manual Emission in Base Case
Replace the call to `_pattern!` with inline code that:
1. Calls the existing 1-D three-phase logic
2. For each column in col[1], manually writes rows to rowval
3. Updates `colptr` with **global** column index = `col_idx + (c - first(col[1]))`

## Files Modified

- `src/stencil.jl`: 
  - New N-D overloads: `assemble(st::LinearStencil{D, O, L, T, N}, row::NTuple{N}, col::NTuple{N})`
  - New: `_pattern_nd!`, `_pattern_nd_recursive`
  - New: `_fill_nd!`, `_fill_nd_recursive`
  - Existing 1-D methods unchanged (most-specific dispatch)
- `test/test_stencil.jl`: 2-D tests added and passing; 3-D tests added (partially working)

## Progress Update (2026-05-18)

### Completed
- **1-D**: Full implementation with 31+ assertions passing
- **2-D**: D=1 and D=2 fully working (2/2 tests passing)
  - Row and column indices correct
  - Coefficient values correct
  - CSC sortedness guaranteed

### 3-D Status
- **D=1**: Partially working
  - Coefficient values correct (fill phase working)
  - Row indices incorrect (pattern phase doesn't account for multiple outer dimensions)
  - Issue: `row_offset = (outer_col_idx - 1) * length(row[1])` only handles 2-D
  - Needs: Row offset to account for strides from dimensions 2 and 3
- **D=2, D=3**: Need investigation

### Root Cause: Pattern Phase Recursion
The pattern phase currently threads a single `outer_col_idx` (scalar) through the recursion, which works for 2-D but breaks for 3-D and higher. The issue manifests when computing row offsets in the base case:

```julia
adjusted_row_offset = (adjusted_outer_col_idx - 1) * length(row[1])
```

For N-D, this should be:
```julia
adjusted_row_offset = sum((c_i - 1) * stride_i for i in 2:N)
```

where `stride_i = prod(length(row[j]) for j in 1:i-1)`.

### Next Steps
1. Refactor pattern phase to thread `outer_coords` tuple instead of scalar `outer_col_idx`
2. Update all base cases (D==1 and D>N_dims) to work with coordinate tuples
3. Implement proper N-D row offset computation
4. Validate with oracle on all 3-D cases

## References

- Plan: `/Users/lechena/.claude/plans/plan-the-implementation-of-gleaming-puzzle.md`
- Oracle: `test/reference.jl` — `stencil_reference(offsets::NTuple{K,CartesianIndex{N}}, ...)`
- Inspiration: `~/.julia/dev/CartesianRuns/src/construction.jl` — `_build_fused!` recursive pattern
