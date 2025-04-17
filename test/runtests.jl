using Test
using ClimaCartesianIndices
using ClimaCartesianIndices: FastCartesianIndices
using Aqua

const DIMS = map(Int32, (4, 4, 1, 63, 5400))
const sci = FastCartesianIndices(DIMS)
const ci = CartesianIndices(DIMS)

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
    sci = FastCartesianIndices(map(Int32, (4, 4, 1, 63, 5400)))
    @test sci.mi[1] isa
          Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int32}

    sci = FastCartesianIndices(map(Int64, (4, 4, 1, 63, 5400)))
    @test sci.mi[1] isa
          Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}
end

@testset "Aqua" begin
    Aqua.test_all(ClimaCartesianIndices)
end
