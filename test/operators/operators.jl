using RadialBasisFunctions
import RadialBasisFunctions as RBF
using StaticArraysCore
using LinearAlgebra
using Statistics
using HaltonSequences

N = 100
x = SVector{2}.(HaltonPoint(2)[1:N])

@testset "Base Methods" begin
    ∂ = partial(x, 1, 1)
    update_weights!(∂)
    @test is_cache_valid(∂)
    RBF.invalidate_cache!(∂)
    @test !is_cache_valid(∂)
end

@testset "Operator Evaluation" begin
    ∂ = partial(x, 1, 1)
    y = rand(N)
    z = rand(N)
    ∂(y, z)
    @test y ≈ ∂.weights * z

    ∇ = gradient(x, PHS(3; poly_deg=2))
    y = (rand(N), rand(N))
    ∇(y, z)
    @test y[1] ≈ ∇.weights[1] * z
    @test y[2] ≈ ∇.weights[2] * z

    @test ∇ ⋅ z ≈ (∇.weights[1] * z) .+ (∇.weights[2] * z)
end

@testset "Printing" begin
    ∂ = partial(x, 1, 1)
    @test repr(∂) == """
    RadialBasisOperator
    ├─Operator: ∂ⁿf/∂xᵢ (n = 1, i = 1)
    ├─Data type: StaticArraysCore.SVector{2, Float64}
    ├─Number of points: 100
    ├─Stencil size: 12
    └─Basis: Polyharmonic spline (r³) with degree 2 polynomial augmentation
    """

    @test RBF.print_op(∂.ℒ) == "∂ⁿf/∂xᵢ (n = 1, i = 1)"
end
