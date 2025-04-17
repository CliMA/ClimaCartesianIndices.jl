# ClimaCartesianIndices.jl

This package implements a type called `FastCartesianIndices`, which is a drop-in replacement for `CartesianIndices`. `FastCartesianIndices` uses Julia Base's
`SignedMultiplicativeInverse` to avoid integer division when converting
linear indexes to cartesian indexes. This is especially useful on the gpu,
which can have a nearly 2x performance impact.

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
function perf_cart_index_fast!(X, Y)
    x1 = X.x1;
    nitems = length(parent(x1));
    max_threads = 256; # can be higher if conditions permit
    nthreads = min(max_threads, nitems);
    nblocks = cld(nitems, nthreads);
    # CI = CartesianIndices(size(x1))
    CI = FastCartesianIndices(map(Int32, size(x1)));
    CUDA.@cuda threads=nthreads blocks=nblocks name="cartesian" perf_cart_index_kernel!(
        X,
        Y,
        Val(nitems),
        Val(CI),
    )
end;
function perf_cart_index!(X, Y)
    x1 = X.x1;
    nitems = length(parent(x1));
    max_threads = 256; # can be higher if conditions permit
    nthreads = min(max_threads, nitems);
    nblocks = cld(nitems, nthreads);
    # CI = CartesianIndices(size(x1))
    CI = CartesianIndices(size(x1));
    CUDA.@cuda threads=nthreads blocks=nblocks name="cartesian" perf_cart_index_kernel!(
        X,
        Y,
        Val(nitems),
        Val(CI),
    )
end;
function perf_cart_index_kernel!(X, Y, ::Val{nitems}, ::Val{CI}) where {nitems, CI}
    (; x1, x2, x3, x4) = X
    (; y1) = Y
    @inbounds begin
        _i = CUDA.threadIdx().x +
          (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x
        if _i ≤ nitems
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

CUDA.@profile begin
    perf_linear_index!(X, Y)
    perf_linear_index!(X, Y)
    perf_linear_index!(X, Y)
    perf_linear_index!(X, Y)
end

CUDA.@profile begin
    perf_cart_index!(X, Y)
    perf_cart_index!(X, Y)
    perf_cart_index!(X, Y)
    perf_cart_index!(X, Y)
end

CUDA.@profile begin
    perf_cart_index_fast!(X, Y)
    perf_cart_index_fast!(X, Y)
    perf_cart_index_fast!(X, Y)
    perf_cart_index_fast!(X, Y)
end

```