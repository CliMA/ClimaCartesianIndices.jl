# ClimaCartesianIndices.jl

This package implements a type called `FastCartesianIndices`, which is a drop-in replacement for `CartesianIndices`. `FastCartesianIndices` uses Julia Base's
`SignedMultiplicativeInverse` to avoid integer division when converting
linear indexes to cartesian indexes. This is especially useful on the gpu,
which can have a nearly 2x performance impact.

!!! warn

	`FastCartesianIndices` internally uses `Int32` and is therefore only
	valid when the product of the input indices are less than or equal to
	`typemax(Int32)` (2147483647)

## Example

```julia
using ClimaCartesianIndices: FastCartesianIndices
using CUDA;

function perf_linear_index!(X, Y)
    x1 = X.x1;
    nitems = length(parent(x1));
    max_threads = 256; # can be higher if conditions permit
    nthreads = min(max_threads, nitems);
    nblocks = cld(nitems, nthreads);
    CUDA.@cuda threads=nthreads blocks=nblocks name="linear" perf_linear_index_kernel!(
        X,
        Y,
        Val(nitems),
    )
end;
function perf_linear_index_kernel!(X, Y, ::Val{nitems}) where {nitems}
    (; x1, x2, x3, x4) = X
    (; y1) = Y
    @inbounds begin
        i = CUDA.threadIdx().x +
          (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x
        if i ≤ nitems
            # more flops makes them much closer
            # y1[i] = x1[i] + x2[i] + x3[i] + x4[i]
            y1[i] = x1[i]
        end
    end
    return nothing
end;
function perf_cart_index!(X, Y, cart_inds)
    x1 = X.x1;
    nitems = length(parent(x1));
    max_threads = 256; # can be higher if conditions permit
    nthreads = min(max_threads, nitems);
    nblocks = cld(nitems, nthreads);
    CUDA.@cuda threads=nthreads blocks=nblocks name="cartesian" perf_cart_index_kernel!(
        X,
        Y,
        Val(nitems),
        cart_inds,
    )
end;
unval(::Val{CI}) where {CI} = CI
unval(CI) = CI
function perf_cart_index_kernel!(X, Y, ::Val{nitems}, cart_inds) where {nitems}
    (; x1, x2, x3, x4) = X
    (; y1) = Y
    @inbounds begin
        _i = CUDA.threadIdx().x +
          (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x
        if _i ≤ nitems && _i ≤ length(unval(cart_inds))
            CI = unval(cart_inds)
            i = CI[_i]
            # more flops makes them much closer
            # y1[i] = x1[i] + x2[i] + x3[i] + x4[i]
            y1[i] = x1[i]
        end
    end
    return nothing
end;

function get_arrays(sym, AType, FT, s, n = 4)
    println("array_type = $AType")
    fn = ntuple(i -> Symbol(sym, i), n)
    return (; zip(fn, ntuple(_ -> AType(zeros(FT, s...)), n))...)
end;
using CUDA;
array_size = (50, 5, 5, 6, 5400); # array
X = get_arrays(:x, CUDA.CuArray, Float64, array_size);
Y = get_arrays(:y, CUDA.CuArray, Float64, array_size);

fast_ci(x) = FastCartesianIndices(map(Int32, size(x)))

CUDA.@profile begin
    perf_linear_index!(X, Y)
    perf_linear_index!(X, Y)
    perf_linear_index!(X, Y)
    perf_linear_index!(X, Y)
end

CUDA.@profile begin
    perf_cart_index!(X, Y, CartesianIndices(X.x1))
    perf_cart_index!(X, Y, CartesianIndices(X.x1))
    perf_cart_index!(X, Y, CartesianIndices(X.x1))
    perf_cart_index!(X, Y, CartesianIndices(X.x1))
end

CUDA.@profile begin
    perf_cart_index!(X, Y, Val(CartesianIndices(X.x1)))
    perf_cart_index!(X, Y, Val(CartesianIndices(X.x1)))
    perf_cart_index!(X, Y, Val(CartesianIndices(X.x1)))
    perf_cart_index!(X, Y, Val(CartesianIndices(X.x1)))
end

CUDA.@profile begin
    perf_cart_index!(X, Y, fast_ci(X.x1))
    perf_cart_index!(X, Y, fast_ci(X.x1))
    perf_cart_index!(X, Y, fast_ci(X.x1))
    perf_cart_index!(X, Y, fast_ci(X.x1))
end

CUDA.@profile begin
    perf_cart_index!(X, Y, Val(fast_ci(X.x1)))
    perf_cart_index!(X, Y, Val(fast_ci(X.x1)))
    perf_cart_index!(X, Y, Val(fast_ci(X.x1)))
    perf_cart_index!(X, Y, Val(fast_ci(X.x1)))
end
```

## Results (NVIDIA A100)

For `perf_linear_index!`:
```julia
Device-side activity: GPU was busy for 1.51 ms (94.36% of the trace)
┌──────────┬────────────┬───────┬──────────────────────────────────────┬────────┐
│ Time (%) │ Total time │ Calls │ Time distribution                    │ Name   │
├──────────┼────────────┼───────┼──────────────────────────────────────┼────────┤
│   94.36% │    1.51 ms │     4 │ 377.24 µs ± 1.5    (375.75 ‥ 379.32) │ linear │
└──────────┴────────────┴───────┴──────────────────────────────────────┴────────┘
```

For `perf_cart_index!(X, Y, CartesianIndices(...))`
```julia
Device-side activity: GPU was busy for 3.56 ms (97.21% of the trace)
┌──────────┬────────────┬───────┬─────────────────────────────────────┬───────────┐
│ Time (%) │ Total time │ Calls │ Time distribution                   │ Name      │
├──────────┼────────────┼───────┼─────────────────────────────────────┼───────────┤
│   97.21% │    3.56 ms │     4 │ 889.06 µs ± 0.19   (888.82 ‥ 889.3) │ cartesian │
└──────────┴────────────┴───────┴─────────────────────────────────────┴───────────┘
```

For `perf_cart_index!(X, Y, Val(CartesianIndices(...)))`
```julia
Device-side activity: GPU was busy for 2.61 ms (95.98% of the trace)
┌──────────┬────────────┬───────┬──────────────────────────────────────┬───────────┐
│ Time (%) │ Total time │ Calls │ Time distribution                    │ Name      │
├──────────┼────────────┼───────┼──────────────────────────────────────┼───────────┤
│   95.98% │    2.61 ms │     4 │ 652.13 µs ± 0.23   (651.84 ‥ 652.31) │ cartesian │
└──────────┴────────────┴───────┴──────────────────────────────────────┴───────────┘
```

For `perf_cart_index!(X, Y, fast_ci(...))`
```julia
Device-side activity: GPU was busy for 2.43 ms (95.79% of the trace)
┌──────────┬────────────┬───────┬──────────────────────────────────────┬───────────┐
│ Time (%) │ Total time │ Calls │ Time distribution                    │ Name      │
├──────────┼────────────┼───────┼──────────────────────────────────────┼───────────┤
│   95.79% │    2.43 ms │     4 │ 607.31 µs ± 0.41   (606.78 ‥ 607.73) │ cartesian │
└──────────┴────────────┴───────┴──────────────────────────────────────┴───────────┘
```

For `perf_cart_index!(X, Y, Val(fast_ci(...)))`
```julia
Device-side activity: GPU was busy for 1.64 ms (91.04% of the trace)
┌──────────┬────────────┬───────┬──────────────────────────────────────┬───────────┐
│ Time (%) │ Total time │ Calls │ Time distribution                    │ Name      │
├──────────┼────────────┼───────┼──────────────────────────────────────┼───────────┤
│   91.04% │    1.64 ms │     4 │ 408.89 µs ± 9.92   (397.21 ‥ 420.09) │ cartesian │
└──────────┴────────────┴───────┴──────────────────────────────────────┴───────────┘
```
