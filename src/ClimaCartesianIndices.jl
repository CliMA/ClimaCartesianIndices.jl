module ClimaCartesianIndices

@inline SMI(x) = Base.MultiplicativeInverses.SignedMultiplicativeInverse(x)

"""
    FastCartesianIndices

`FastCartesianIndices` is a drop-in replacement for `CartesianIndices`.

Unlike `CartesianIndices`, `FastCartesianIndices` avoids integer
division by using `Base.MultiplicativeInverses.SignedMultiplicativeInverse`.

!!! warn

    `FastCartesianIndices` internally uses `Int32` and is therefore only
    valid when the product of the input indices are less than or equal to
    `typemax(Int32)` (2147483647)
"""
struct FastCartesianIndices{N, MI} <: AbstractArray{CartesianIndex{N}, N}
    mi::MI
end
FastCartesianIndices(::Tuple{}) = FastCartesianIndices{0, ()}(())
function FastCartesianIndices(inds::NTuple)
    N = length(inds)
    size_inds = if eltype(inds) <: Integer
        inds
    elseif eltype(inds) <: Union{UnitRange, Base.OneTo}
        map(i -> i.stop, inds)
    end
    mi = map(SMI, size_inds)
    @assert size_inds isa Tuple{Vararg{T}} where {T <: Integer} # need this for getindex
    return FastCartesianIndices{N, typeof(mi)}(mi)
end
FastCartesianIndices(x::AbstractArray) = FastCartesianIndices(axes(x))

Base.:(==)(a::FastCartesianIndices{N}, b::FastCartesianIndices{N}) where {N} =
    all(map(==, a.mi, b.mi))
Base.:(==)(a::FastCartesianIndices, b::FastCartesianIndices) = false

Base.size(iter::FastCartesianIndices) = map(x -> x.divisor, iter.mi)
Base.length(iter::FastCartesianIndices) = prod(size(iter))
Base.axes(iter::FastCartesianIndices) = map(x -> Base.OneTo(x.divisor), iter.mi)
__tail(iter::FastCartesianIndices{N}) where {N} =
    FastCartesianIndices{N - 1, typeof(inds_tail)}(Base.tail(iter.mi))

Base.@propagate_inbounds Base.getindex(iter::FastCartesianIndices{0}) =
    CartesianIndex()
@inline function Base.getindex(
    iter::FastCartesianIndices{N},
    I::Vararg{<:Integer, N},
) where {N}  # exit point
    # Eagerly do boundscheck before calculating each item of the CartesianIndex so that
    # we can pass `@inbounds` hint to inside the map and generates more efficient SIMD codes (#42115)
    @boundscheck Base.checkbounds_indices(Bool, axes(iter), I)
    divisors = map(x -> x.divisor, iter.mi)
    index = map(divisors, I) do r, i
        @inbounds getindex(1:r, i)
    end
    CartesianIndex(index)
end
function Base.getindex(A::FastCartesianIndices, I...) # entry point
    Base.@_propagate_inbounds_meta
    Base.error_if_canonical_getindex(IndexStyle(A), A, I...)
    _getindex(A, _to_indices(A, (), I)...)
end
_to_indices(A, inds, ::Tuple{}) = ()

to_index(A, i) = to_index(i)
_to_indices1(A::FastCartesianIndices, inds, I1) = (I1,)
_to_indices1(
    A::FastCartesianIndices,
    inds,
    I1::FastCartesianIndices{N},
) where {N} = map(y -> to_index(A, 1:(y.divisor)), A.mi)
_cutdim(inds, I1) = safe_tail(inds)
safe_tail(t::Tuple) = Base.tail(t)
safe_tail(t::Tuple{}) = ()

# but preserve FastCartesianIndices{0} as they consume a dimension.
_to_indices1(A, inds, I1::FastCartesianIndices{0}) = (I1,)
_to_indices1(A::FastCartesianIndices, inds, I1::FastCartesianIndices{0}) = (I1,)

function _to_indices(A, inds, I::Tuple{Any, Vararg{Any}})
    @inline
    head = _to_indices1(A, inds, I[1])
    rest = _to_indices(A, _cutdim(inds, I[1]), Base.tail(I))
    (head..., rest...)
end

function _getindex(
    A::FastCartesianIndices{N},
    I::Vararg{<:Integer, M},
) where {N, M}
    @inline
    @boundscheck Base.checkbounds_indices(Bool, axes(A), I) # generally _to_subscript_indices requires bounds checking
    @inbounds r = Base.getindex(A, _to_subscript_indices(A, I...)...)
    r
end

_to_subscript_indices(A::FastCartesianIndices, i::Integer) =
    (@inline; _unsafe_ind2sub(A, i))
_to_subscript_indices(
    A::FastCartesianIndices{N},
    I::Vararg{<:Integer, N},
) where {N} = I
_to_subscript_indices(A::FastCartesianIndices{0}, i::Integer) = ()
_to_subscript_indices(A::FastCartesianIndices{0}, I::Integer...) = ()

_unsafe_ind2sub(::Tuple{}, i) = () # _ind2sub may throw(BoundsError()) in this case
_unsafe_ind2sub(sz, i) = (@inline; _ind2sub(sz, i))

_ind2sub(inds::FastCartesianIndices, ind::Integer) = (@inline;
@inbounds _ind2sub_recurse(size(inds), inds, ind - Int32(1), Int32(1)))

@inline _ind2sub_recurse(::Tuple{}, ::FastCartesianIndices, ind, n) =
    (ind + Int32(1),)
@inline _ind2sub_recurse(
    ::Tuple{<:Integer},
    indslast::FastCartesianIndices,
    ind,
    n,
) = (ind + Int32(1),)
@inline function _ind2sub_recurse(
    t,
    inds::FastCartesianIndices{N},
    ind,
    n,
) where {N}
    r1 = inds.mi[n].divisor
    (; mi) = inds
    _mi = mi[n]
    indnext, l = div(Signed(Int32(ind)), _mi), r1 # on julia-side avoids div(,Int)
    (
        ind - l * indnext + 1,
        _ind2sub_recurse(Base.tail(t), inds, indnext, n + 1)...,
    )
end

function Base.show(io::IO, iter::FastCartesianIndices)
    print(io, "FastCartesianIndices(")
    show(io, map(_xform_index, map(x -> Base.OneTo(x), size(iter))))
    print(io, ")")
end
_xform_index(i) = i
_xform_index(i::Base.OneTo) = i.stop
Base.show(io::IO, ::MIME"text/plain", iter::FastCartesianIndices) =
    show(io, iter)

end # module ClimaCartesianIndices
