"""
    Directional <: ScalarValuedOperator

Operator for the directional derivative, or the inner product of the gradient and a direction vector.
"""
struct Directional{L<:NTuple,T} <: ScalarValuedOperator
    ℒ::L
    v::T
end

"""
    function directional(data, v, basis; k=autoselect_k(data, basis))

Builds a `RadialBasisOperator` where the operator is the directional derivative, `Directional`.
"""
function directional(
    data::AbstractVector,
    v::AbstractVector,
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, k),
) where {B<:AbstractRadialBasis,T<:Int}
    f = ntuple(dim -> Base.Fix2(∂, dim), length(first(data)))
    ℒ = Directional(f, v)
    return RadialBasisOperator(ℒ, data, basis; k=k, adjl=adjl)
end

"""
    function directional(data, eval_points, v, basis; k=autoselect_k(data, basis))

Builds a `RadialBasisOperator` where the operator is the directional derivative, `Directional`.
"""
function directional(
    data::AbstractVector,
    eval_points::AbstractVector,
    v::AbstractVector,
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, eval_points, k),
) where {B<:AbstractRadialBasis,T<:Int}
    f = ntuple(dim -> Base.Fix2(∂, dim), length(first(data)))
    ℒ = Directional(f, v)
    return RadialBasisOperator(ℒ, data, eval_points, basis; k=k, adjl=adjl)
end

function RadialBasisOperator(
    ℒ::Directional,
    data::AbstractVector,
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, k),
) where {T<:Int,B<:AbstractRadialBasis}
    weights = _build_weights(ℒ, data, data, adjl, basis)
    return RadialBasisOperator(ℒ, weights, data, data, adjl, basis)
end

function RadialBasisOperator(
    ℒ::Directional,
    data::AbstractVector{TD},
    eval_points::AbstractVector{TE},
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, eval_points, k),
) where {TD,TE,T<:Int,B<:AbstractRadialBasis}
    weights = _build_weights(ℒ, data, eval_points, adjl, basis)
    return RadialBasisOperator(ℒ, weights, data, eval_points, adjl, basis)
end

function _build_weights(ℒ::Directional, data, eval_points, adjl, basis)
    v = ℒ.v
    N = length(first(data))
    @assert length(v) == N || length(v) == length(data) "wrong size for v"
    if length(v) == N
        return mapreduce(+, enumerate(ℒ.ℒ)) do (i, ℒ)
            _build_weights(ℒ, data, eval_points, adjl, basis) * v[i]
        end
    else
        vv = ntuple(i -> getindex.(v, i), N)
        return mapreduce(+, enumerate(ℒ.ℒ)) do (i, ℒ)
            Diagonal(vv[i]) * _build_weights(ℒ, data, eval_points, adjl, basis)
        end
    end
end

function update_weights!(op::RadialBasisOperator{<:Directional})
    v = op.ℒ.v
    N = length(first(op.data))
    if length(v) == N
        op.weights .= mapreduce(+, enumerate(op.ℒ.ℒ)) do (i, ℒ)
            _build_weights(ℒ, op) * v[i]
        end
    else
        vv = ntuple(i -> getindex.(v, i), N)
        op.weights .= mapreduce(+, enumerate(op.ℒ.ℒ)) do (i, ℒ)
            Diagonal(vv[i]) * _build_weights(ℒ, op)
        end
    end
    validate_cache!(op)
    return nothing
end

# pretty printing
print_op(op::Directional) = "Directional Derivative (∇f⋅v)"
