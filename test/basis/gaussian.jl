using RadialBasisFunctions
import RadialBasisFunctions as RBF
using StaticArrays

@testset "Constructors and Printing" begin
    g = Gaussian()
    @test g isa Gaussian
    @test g.ε == 1
    @test g.poly_deg == 2

    g = Gaussian(5.0; poly_deg=0)
    @test g.ε ≈ 5
    @test g.poly_deg == 0

    @test_throws ArgumentError Gaussian(-1)
    @test_throws ArgumentError Gaussian(; poly_deg=-2)

    @test repr(g) == """
    Gaussian, exp(-(ε*r)²)
    ├─Shape factor: ε = 5.0
    └─Polynomial augmentation: degree 0
    """
end

x₁ = SVector(1.0, 2)
x₂ = SVector(2.0, 4)
g = Gaussian(2; poly_deg=-1)

@testset "Distances" begin
    r = sqrt((x₁[1] - x₂[1])^2 + (x₁[2] - x₂[2])^2)
    @test g(x₁, x₂) ≈ exp(-(g.ε * r)^2)
end

@testset "Derivatives" begin
    dim = 1
    ∂rbf = RBF.∂(g, dim)
    ∂²rbf = RBF.∂²(g, dim)
    ∇rbf = RBF.∇(g)

    @test ∂rbf(x₁, x₂) ≈ 8 / exp(20)
    @test all(∇rbf(x₁, x₂) .≈ (8 / exp(20), 16 / exp(20)))
    @test ∂²rbf(x₁, x₂) ≈ 56 / exp(20)
end
