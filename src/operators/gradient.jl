"""
    Gradient <: VectorValuedOperator

Builds an operator for the gradient of a function.
"""
struct Gradient{L<:NTuple} <: VectorValuedOperator
    ℒ::L
end

# convienience constructors
"""
    function gradient(data, basis; k=autoselect_k(data, basis))

Builds a `RadialBasisOperator` where the operator is the gradient, `Gradient`.
"""
function gradient(
    data::AbstractVector,
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, k),
) where {B<:AbstractRadialBasis,T<:Int}
    f = ntuple(dim -> Base.Fix2(∂, dim), length(first(data)))
    ℒ = Gradient(f)
    return RadialBasisOperator(ℒ, data, basis; k=k, adjl=adjl)
end

"""
    function gradient(data, eval_points, basis; k=autoselect_k(data, basis))

Builds a `RadialBasisOperator` where the operator is the gradient, `Gradient`. The resulting operator will only evaluate at `eval_points`.
"""
function gradient(
    data::AbstractVector,
    eval_points::AbstractVector,
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, eval_points, k),
) where {B<:AbstractRadialBasis,T<:Int}
    f = ntuple(dim -> Base.Fix2(∂, dim), length(first(data)))
    ℒ = Gradient(f)
    return RadialBasisOperator(ℒ, data, eval_points, basis; k=k, adjl=adjl)
end

# pretty printing
print_op(op::Gradient) = "Gradient (∇f)"
