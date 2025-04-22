using Test
using JET
using ClimaCartesianIndices
import ClimaCartesianIndices.FastCartesianIndices as FCI
using Aqua

const DIMS = map(Int32, (4, 4, 1, 63, 5400))
const sci = FCI(DIMS)
const ci = CartesianIndices(DIMS)
const SMI{T} =
    Base.MultiplicativeInverses.SignedMultiplicativeInverse{T} where {T}

@testset "ClimaCartesianIndices" begin
    pass_count = 0
    for i in 1:prod(DIMS)
        pass_count += (ci[i] == sci[i])
        pass_count += (ci[Int32(i)] == sci[Int32(i)])
        pass_count += (ci[Int64(i)] == sci[Int64(i)])
    end
    @test pass_count == prod(DIMS) * 3
end

@testset "Type preservation" begin
    sci = FCI(map(Int32, (4, 4, 1, 63, 5400)))
    @test sci.mi[1] isa SMI{Int32}

    sci = FCI(map(Int64, (4, 4, 1, 63, 5400)))
    @test sci.mi[1] isa SMI{Int64}
end

@testset "Interface" begin
    sci = FCI(map(Int32, (4, 4, 1, 63, 5400)))
    @test length(sci) == prod((4, 4, 1, 63, 5400))
    @test size(sci) == (4, 4, 1, 63, 5400)
    arr = rand(3)
    @test size(FCI((1:3, 1:3))) == (3, 3)
    @test FCI(arr) == FCI(size(arr))
    @test FCI((1:3, 1:4)).mi == FCI((3, 4)).mi
    @test FCI((Base.OneTo(3), Base.OneTo(4))).mi == FCI((3, 4)).mi
    @test FCI((Base.OneTo(3), Base.OneTo(4))) == FCI((3, 4))
end

@testset "Type-stability" begin
    sci = FCI(map(Int32, (4, 4, 1, 63, 5400)))
    @test_opt sci[1]
    @test_opt sci[Int32(1)]
end

function foo(s)
    mfc = MyFastCartesianIndices(s)
    mfc[1]
    return nothing
end
function MyFastCartesianIndices(x::Tuple)
    @assert length(CartesianIndices(x)) ≤ typemax(Int32)
    inds = if eltype(x) <: Union{Base.OneTo, UnitRange}
        map(i -> Base.OneTo(Int32(i.stop)), x)
    else
        map(i -> Base.OneTo(Int32(i)), x)
    end
    return FCI(inds)
end

@testset "Type-stability 2" begin
    sci = FCI(map(Int32, (4, 4, 1, 63, 5400)))
    @test_opt sci[1]
    foo((4, 4, 1, 63, 5400)) # make sure this compiles
    @test_opt foo((4, 4, 1, 63, 5400))
    @test_opt sci[Int32(1)]
end

@testset "Aqua" begin
    Aqua.test_all(ClimaCartesianIndices)
end
