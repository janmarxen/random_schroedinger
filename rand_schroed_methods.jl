include("rand_schroed_utils.jl")
using SparseArrays
using LinearAlgebra
using Printf

"""
Solve a linear system Ax = b using the Conjugate Gradient method.

# Arguments
- `A::AbstractMatrix`: Symmetric positive definite matrix.
- `b::AbstractVector`: Right-hand side vector.
- `x0::AbstractVector`: Initial guess for the solution.
- `K::Int`: Maximum number of iterations.
- `min_eps::Float64`: Convergence tolerance.

# Returns
- `x::AbstractVector`: Approximation of the solution vector.
"""
function conjugate_gradient(A, b, x0, K, min_eps)
    x_old = x0
    x = x_old
    r = b - A * x[:, 1]
    p = r
    r_dot = dot(r, r)

    for k in range(2, K)
        alpha = r_dot / dot(p, A * p)
        x = x_old + alpha * p
        eps = sqrt(r_dot)
        # @printf("Iteration %d - eps: %f \n", k, eps)
        if eps <= min_eps
            return x
        end
        r -= alpha * A * p
        old_r_dot = copy(r_dot)
        r_dot = dot(r, r)
        omega = r_dot / old_r_dot
        p = r + omega * p
        x_old = x
    end
    return x
end

"""
Compute the inverse of a symmetric positive definite matrix using the Conjugate Gradient method.

# Arguments
- `A::AbstractMatrix`: Symmetric positive definite matrix.
- `K::Int`: Maximum number of iterations for CG.
- `min_eps::Float64`: Minimum tolerance for CG convergence.

# Returns
- `A_inv::AbstractMatrix`: Approximation of the inverse of `A`.
"""
function inverse_cg(A, K, min_eps)
    N = size(A, 1)
    A_inv = zeros((N, N))
    for i in 1:N
        e_i = zeros(N)
        e_i[i] = 1.0
        x0 = zeros(N)
        A_inv[:, i] = conjugate_gradient(A, e_i, x0, K, min_eps)
    end
    return A_inv
end

"""
Compute the smallest eigenvalue and corresponding eigenvector using the inverse power method.

# Arguments
- `A::AbstractMatrix`: Symmetric positive definite matrix.
- `x0::AbstractVector`: Initial guess for the eigenvector.
- `min_eps::Float64`: Convergence tolerance.

# Returns
- `lambda::Float64`: Smallest eigenvalue.
- `y::AbstractVector`: Corresponding eigenvector (normalized).
"""
function inverse_power_method(A, x0, min_eps)
    K = Int(10e10)  # Care only about the tolerance
    cg_min_eps = min_eps  # Choose a tolerance for CG same as inverse power method
    x = x0
    y = x / norm(x, 2)
    y_old = y
    eps = +Inf

    # i = 0
    lambda = +Inf
    while eps > min_eps
        x = conjugate_gradient(A, y, x, K, cg_min_eps)
        lambda = dot(y, x)
        eps = norm(y_old - lambda * A * y, 2)
        # @printf("Iteration %d - eps: %f \n", i, eps)
        y_old = y
        y = x / norm(x, 2)
        # i += 1
    end

    return 1 / lambda, x / norm(x, 2)
end

"""
Compute the smallest eigenvalue and corresponding eigenvector using the inverse power method,
storing all iterations' eigenvalues and eigenvectors.

# Arguments
- `A::AbstractMatrix`: Symmetric positive definite matrix.
- `x0::AbstractVector`: Initial guess for the eigenvector.
- `min_eps::Float64`: Convergence tolerance.

# Returns
- `eigenvalues::AbstractVector`: Vector of eigenvalues from all iterations.
- `eigenvectors::AbstractMatrix`: Matrix where each column is an eigenvector from all iterations.
"""
function inverse_power_method_store(A, x0, min_eps)
    K = Int(10e10)  # Care only about the tolerance
    cg_min_eps = min_eps  # Choose a tolerance for CG same as inverse power method
    x = x0
    y = x / norm(x, 2)
    y_old = y
    eps = +Inf

    eigenvalues = Float64[]
    eigenvectors = []

    # i = 0
    while eps > min_eps
        x = conjugate_gradient(A, y, x, K, cg_min_eps)
        lambda = dot(y, x)
        push!(eigenvalues, 1/lambda)
        eps = norm(y_old - lambda * A * y, 2)
        # @printf("Iteration %d - eps: %f \n", i, eps)
        y_old = y
        y = x / norm(x, 2)
        push!(eigenvectors, copy(y))
        # i += 1
    end
    eigenvectors = hcat(eigenvectors...)
    return eigenvalues, eigenvectors
end

"""
Compute the largest eigenvalue and corresponding eigenvector using the power method.

# Arguments
- `A::AbstractMatrix`: Symmetric positive definite matrix.
- `x0::AbstractVector`: Initial guess for the eigenvector.
- `min_eps::Float64`: Convergence tolerance.

# Returns
- `lambda::Float64`: Largest eigenvalue.
- `y::AbstractVector`: Corresponding eigenvector (normalized).
"""
function power_method(A, x0, min_eps)
    y = x0
    eps = +Inf
    y = y / norm(y, 2)
    lambda = NaN
    i = 0
    while eps > min_eps
        y_old = y
        y = A * y
        lambda = dot(y_old, y)
        eps = norm(y - lambda * y_old, 2)
        # @printf("Iteration %d - eps: %f \n", i, eps)
        y = y / norm(y, 2)
        i += 1
    end

    return lambda, y
end

"""
Compute the smallest `n_eigen` eigenvalues and their corresponding eigenvectors using deflation and the inverse power method.

# Arguments
- `A::AbstractMatrix`: Symmetric positive definite matrix.
- `x::AbstractVector`: Initial guess for the eigenvector.
- `min_eps::Float64`: Convergence tolerance.
- `n_eigen::Int`: Number of eigenvalues to compute.

# Returns
- `Lambda::AbstractVector`: Vector of smallest eigenvalues.
- `Y::AbstractMatrix`: Matrix where each row is an eigenvector.
"""
# function deflation_min(A, x, min_eps, n_eigen)
#     Lambda = zeros(n_eigen)
#     Y = zeros(n_eigen, size(A, 2))
#     A_deflated = A
#     for n in 1:n_eigen
#         # @printf("Computing %d th lowest eigenvalue... \n", n)
#         Lambda[n], Y[n, :] = inverse_power_method(A_deflated, x, min_eps)
#         y = Y[n, :]
#         deflation_mat = (y * y') / Lambda[n]
#         A_deflated = A_deflated - deflation_mat
#     end
#     return Lambda, Y
# end
function deflation_min(A, x, min_eps, n_eigen)
    Lambda = zeros(n_eigen)
    Y = zeros(n_eigen, size(A, 2))
    A_inv_delfated = inverse_cg(A, 1000, min_eps)
    for n in 1:n_eigen
        # @printf("Computing %d th lowest eigenvalue... \n", n)
        Lambda[n], Y[n, :] = power_method(A_inv_delfated, x, min_eps)
        y = Y[n, :]
        deflation_mat = (y * y') * Lambda[n]
        A_inv_delfated = A_inv_delfated - deflation_mat
    end
    return 1 ./ Lambda, Y
end


"""
Compute the largest `n_eigen` eigenvalues and their corresponding eigenvectors using deflation and the power method.

# Arguments
- `A::AbstractMatrix`: Symmetric positive definite matrix.
- `x::AbstractVector`: Initial guess for the eigenvector.
- `min_eps::Float64`: Convergence tolerance.
- `n_eigen::Int`: Number of eigenvalues to compute.

# Returns
- `Lambda::AbstractVector`: Vector of largest eigenvalues.
- `Y::AbstractMatrix`: Matrix where each row is an eigenvector.
"""
function deflation_max(A, x, min_eps, n_eigen)
    Lambda = zeros(n_eigen)
    Y = zeros(n_eigen, size(A, 2))
    A_deflated = A

    for n in 1:n_eigen
        # @printf("Computing %d th largest eigenvalue... \n", n)
        Lambda[n], Y[n, :] = power_method(A_deflated, x, min_eps)
        y = Y[n, :]
        deflation_mat = (y * y') / Lambda[n]
        A_deflated = A_deflated - deflation_mat
    end
    return Lambda, Y
end


function conjugate_gradient_2D(L, v, w, b, x0, K, min_eps)
    """
    Conjugate Gradient solver for (L(2) + v kronecker w) x = b,
    adjusted for matrix-like residuals (r, p, etc.) directly without flattening.
    
    Arguments:
    - `L::SparseMatrixCSC`: The 1D finite-difference Laplacian matrix.
    - `v::Vector`: Potential vector v.
    - `w::Vector`: Potential vector w.
    - `b::Matrix`: Right-hand side matrix (nxn).
    - `x0::Matrix`: Initial guess (nxn).
    - `K::Int`: Maximum number of iterations.
    - `min_eps::Float64`: Convergence tolerance.

    Returns:
    - Approximate solution matrix `x`.
    """
    x_old = x0
    r = b - Hmatvec(L, v, w, x_old)  # Compute initial residual (r is nxn)
    p = r  
    r_dot = sum(r .* r)  # Compute the Frobenius norm squared (matrix inner product)
    for k in 1:K
        # Compute the H matrix-vector product
        Hp = Hmatvec(L, v, w, p)  # Compute H(p), same as matrix-vector multiplication
        alpha = r_dot / sum(p .* Hp)  # Scale using Frobenius inner product
        # Update solution vector/matrix
        x = x_old + alpha * p
        eps = sqrt(r_dot)  
        if eps <= min_eps
            # @printf("CG converged with eps=%f and %d iterations... \n", eps, k)
            return x
        end
        r -= alpha * Hp
        old_r_dot = copy(r_dot)
        r_dot = sum(r .* r)  # Update residual Frobenius norm squared
        # Update conjugate direction with omega
        omega = r_dot / old_r_dot
        p = r + omega * p
        x_old = x
    end
    return x
end

"""
Compute the inverse of a symmetric positive definite matrix using the Conjugate Gradient method.

# Arguments
- `A::AbstractMatrix`: Symmetric positive definite matrix.
- `K::Int`: Maximum number of iterations for CG.
- `min_eps::Float64`: Minimum tolerance for CG convergence.

# Returns
- `A_inv::AbstractMatrix`: Approximation of the inverse of `A`.
"""
function inverse_cg_2D(L, v, w, K, min_eps)
    n = size(L, 1)
    n_pow_2 = n^2
    L2_inv = zeros((n_pow_2, n_pow_2))
    for i in 1:n_pow_2
        e_i = zeros(n, n)
        e_i[i] = 1.0
        x0 = zeros(n, n)
        L2_inv[:, i] = vec(conjugate_gradient_2D(L, v, w, e_i, x0, K, min_eps))
    end
    return L2_inv
end

function deflation_min_2D(L, v, w, x, min_eps, n_eigen)
    Lambda = zeros(n_eigen)
    n_pow_2 = size(L, 1)^2
    Y = zeros(n_eigen, n_pow_2)
    L2_inv_delfated = inverse_cg_2D(L, v, w, 1000, min_eps)
    x = vec(x)
    for n in 1:n_eigen
        # @printf("Computing %d th lowest eigenvalue... \n", n)
        Lambda[n], Y[n, :] = power_method(L2_inv_delfated, x, min_eps)
        y = Y[n, :]
        deflation_mat = (y * y') * Lambda[n]
        L2_inv_delfated = L2_inv_delfated - deflation_mat
    end
    return 1 ./ Lambda, Y
end
