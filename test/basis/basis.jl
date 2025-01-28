using RadialBasisFunctions
import RadialBasisFunctions as RBF

@testset "Pretty Printing" begin
    superscripts = ("", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹")
    @test all(unicode_order.(Val.(1:9)) .== superscripts)
end

@testset "Basis - General Utils" begin
    m = MonomialBasis(2, 3)
    @test dim(m) == 2
    @test degree(m) == 3

    ∂m = RBF.∂(m, 1)
    @test dim(∂m) == 2
    @test degree(∂m) == 3
end
