using RadialBasisFunctions

@test_throws ArgumentError RadialBasisFunctions.ℒMonomialBasis(1, -1, identity)
m = RadialBasisFunctions.ℒMonomialBasis(1, 0, identity)
@test repr(m) == "ℒMonomialBasis of degree 0 in 1 dimensions"
