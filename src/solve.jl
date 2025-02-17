function _build_weights(ℒ, op)
    data = op.data
    eval_points = op.eval_points
    adjl = op.adjl
    basis = op.basis
    return _build_weights(ℒ, data, eval_points, adjl, basis)
end

function _build_weights(ℒ, data, eval_points, adjl, basis)
    dim = length(first(data)) # dimension of data

    # build monomial basis and operator
    mon = MonomialBasis(dim, basis.poly_deg)
    ℒmon = ℒ(mon)
    ℒrbf = ℒ(basis)

    return _build_weights(data, eval_points, adjl, basis, ℒrbf, ℒmon, mon)
end

function _build_weights(data, eval_points, adjl, basis, ℒrbf, ℒmon, mon)
    nchunks = Threads.nthreads()
    TD = eltype(first(data))
    dim = length(first(data)) # dimension of data
    nmon = binomial(dim + basis.poly_deg, basis.poly_deg)
    k = length(first(adjl))  # number of data in influence/support domain
    sizes = (k, nmon)

    # allocate arrays to build sparse matrix
    Na = length(adjl)
    I = zeros(Int, k * Na)
    J = reduce(vcat, adjl)
    V = zeros(TD, k * Na)

    # create work arrays
    n = sum(sizes)
    A = Symmetric[Symmetric(zeros(TD, n, n), :U) for _ in 1:nchunks]
    b = Vector[zeros(TD, n) for _ in 1:nchunks]
    d = Vector{Vector{eltype(data)}}(undef, nchunks)

    # build stencil for each data point and store in global weight matrix
    Threads.@threads for (ichunk, xrange) in enumerate(index_chunks(adjl; n=nchunks))
        for i in xrange
            I[((i - 1) * k + 1):(i * k)] .= i
            d[ichunk] = data[adjl[i]]
            V[((i - 1) * k + 1):(i * k)] = @views _build_stencil!(
                A[ichunk], b[ichunk], ℒrbf, ℒmon, d[ichunk], eval_points[i], basis, mon, k
            )
        end
    end

    return sparse(I, J, V, length(adjl), length(data))
end

function _build_stencil!(
    A::Symmetric,
    b::Vector,
    ℒrbf,
    ℒmon,
    data::AbstractVector{TD},
    eval_point::TE,
    basis::B,
    mon::MonomialBasis,
    k::Int,
) where {TD,TE,B<:AbstractRadialBasis}
    _build_collocation_matrix!(A, data, basis, mon, k)
    _build_rhs!(b, ℒrbf, ℒmon, data, eval_point, basis, k)
    return (A \ b)[1:k]
end

function _build_collocation_matrix!(
    A::Symmetric, data::AbstractVector, basis::B, mon::MonomialBasis{Dim,Deg}, k::K
) where {B<:AbstractRadialBasis,K<:Int,Dim,Deg}
    # radial basis section
    AA = parent(A)
    N = size(A, 2)
    @inbounds for j in 1:k, i in 1:j
        AA[i, j] = basis(data[i], data[j])
    end

    # monomial augmentation
    if Deg > -1
        @inbounds for i in 1:k
            a = view(AA, i, (k + 1):N)
            mon(a, data[i])
        end
    end

    return nothing
end

function _build_collocation_matrix_Hermite!(
    A::Symmetric, data::AbstractVector, data_info, basis::B, mon::MonomialBasis{Dim,Deg}, k::K
) where {B<:AbstractRadialBasis,K<:Int,Dim,Deg}
    # radial basis section
    AA = parent(A)
    N = size(A, 2)
    @inbounds for j in 1:k, i in 1:j
        _calculate_matrix_entry!(AA, i, j, data, data_info, basis)
    end

    # monomial augmentation
    if Deg > -1
        @inbounds for i in 1:k
            a = view(AA, i, (k + 1):N)
            mon(a, data[i])
        end
    end

    return nothing
end

function _calculate_matrix_entry!(A, i, j, data, data_info, basis)
    is_Neumann_i = data_info.is_Neumann[i]
    is_Neumann_j = data_info.is_Neumann[j]
    if !is_Neumann_i && !is_Neumann_j
        A[i, j] = basis(data[i], data[j])
    elseif is_Neumann_i && !is_Neumann_j
        n = data_info.normal[i]
        A[i, j] = directional∂(basis,n)(data[i], data[j])
    elseif !is_Neumann_i && is_Neumann_j
        n = data_info.normal[j]
        A[i, j] = directional∂(basis,n,Hermite=true)(data[i], data[j])
    elseif is_Neumann_i && is_Neumann_j
        ni = data_info.normal[i]
        nj = data_info.normal[j]
        A[i, j] = directional∂²(basis, ni, nj)(data[i], data[j])
    end
    return nothing
end

function _build_rhs!(
    b::AbstractVector, ℒrbf, ℒmon, data::AbstractVector{TD}, eval_point::TE, basis::B, k::K
) where {TD,TE,B<:AbstractRadialBasis,K<:Int}
    # radial basis section
    @inbounds for i in eachindex(data)
        b[i] = ℒrbf(eval_point, data[i])
    end

    # monomial augmentation
    if basis.poly_deg > -1
        N = length(b)
        bmono = view(b, (k + 1):N)
        ℒmon(bmono, eval_point)
    end

    return nothing
end

function _build_rhs!(
    b::AbstractVector, ℒrbf, ℒmon, data::AbstractVector{TD}, data_info, eval_point::TE, basis::B, k::K
) where {TD,TE,B<:AbstractRadialBasis,K<:Int}
    # radial basis section
    @inbounds for i in eachindex(data)
        b[i] = ℒrbf(eval_point, data[i], Hermite=data_info.is_Neumann[i])
    end

    # monomial augmentation
    if basis.poly_deg > -1
        N = length(b)
        bmono = view(b, (k + 1):N)
        ℒmon(bmono, eval_point)
    end

    return nothing
end