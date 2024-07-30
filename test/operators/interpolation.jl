using RadialBasisFunctions
using StaticArrays
using Random

"""
    franke(x)

Popular test function for interpolation. Franke, R. (1979). A critical comparison of some methods for interpolation of scattered data (No. NPS53-79-003). NAVAL POSTGRADUATE SCHOOL MONTEREY CA.
"""
function franke(x)
    a = 0.75 * exp(-(9x[1] - 2)^2 / 4 - (9x[2] - 2)^2 / 4)
    b = 0.75 * exp(-(9x[1] + 1)^2 / 49 - (9x[2] + 1) / 10)
    c = 0.5 * exp(-(9x[1] - 7)^2 / 4 - (9x[2] - 3)^2 / 4)
    d = 0.2 * exp(-(9x[1] - 4)^2 - (9x[2] - 7)^2)
    return a + b + c - d
end

N = 100
Δ = 1 / (N - 1)
points = 0:Δ:1
structured_points = ((x, y) for x in points for y in points)
x = map(x -> SVector{2}(x .+ (Δ / 2 .* rand(2))), structured_points)
y = franke.(x)

interp = Interpolator(x, y, PHS(3; poly_deg=2))
@test interp isa Interpolator

xnew = SVector(0.5, 0.5)
@test abs(interp(xnew) - franke(xnew)) < 1e-5
