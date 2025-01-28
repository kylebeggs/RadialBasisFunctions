using RadialBasisFunctions
using StaticArrays
using Statistics
using HaltonSequences

rsme(test, correct) = sqrt(sum((test - correct) .^ 2) / sum(correct .^ 2))
mean_percent_error(test, correct) = mean(abs.((test .- correct) ./ correct)) * 100

f(x) = 1 + sin(4 * x[1]) + cos(3 * x[1]) + sin(2 * x[2])

N = 10_000
x = SVector{2}.(HaltonPoint(2)[1:N])
y = f.(x)

x2 = map(x -> SVector{2}(rand(2)), 1:100)
r = regrid(x, x2, PHS(3; poly_deg=2))
@test mean_percent_error(r(y), f.(x2)) < 0.1

@test RadialBasisFunctions.print_op(r.ℒ) == "regrid"
