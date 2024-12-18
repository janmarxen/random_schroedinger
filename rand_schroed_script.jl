include("rand_schroed_utils.jl")
include("rand_schroed_methods.jl")

using SparseArrays
using LinearAlgebra
using Plots
using Printf

function ex1(n)
    return rand_schrodinger_1d(n)
end

function ex2(Lpv, n, K, min_eps)
    x0 = randn(n)
    b = ones(n)
    x_sol = conjugate_gradient(Lpv, b, x0, K, min_eps)
    ground_truth = Lpv \ b
    comparison = norm(x_sol - ground_truth, 2)
    return x_sol, comparison
end

function ex3(Lpv, ground_truth_eigenvals, min_eps)
    min_lambda, Y = inverse_power_method(Lpv, randn(size(Lpv,1)), min_eps)
    comparison = abs(min_lambda-minimum(ground_truth_eigenvals))
    return comparison
end

function ex4(Lpv, n, min_eps, n_eigen, ground_truth_eigenvals)
    x = randn(n)
    Lambda, Y = deflation_min(Lpv, x, min_eps, n_eigen)
    comparison = norm(sort(Lambda)-ground_truth_eigenvals[1:n_eigen], 2)
    return Lambda, Y, comparison
end

function ex5(Lpv, x0, min_eps)
    F = eigen(Array(Lpv))
    idx = sortperm(F.values)
    ground_truth_eigenvals = F.values
    ground_truth_eigenvecs = F.vectors
    # Smallest eigenvalue approximation (storing iterations on the way)
    lambda_iters, y_iters = inverse_power_method_store(Lpv, x0, min_eps);
    eigenvalue_differences = [abs(lambda_iters[i] - ground_truth_eigenvals[idx[1]]) for i in 1:length(lambda_iters)]
    eigenvector_differences = [norm(y_iters[:, i] - ground_truth_eigenvecs[:, idx[1]], 2) for i in 1:size(y_iters, 2)]
    lambda_1_o_2 = [(ground_truth_eigenvals[idx[1]]/ground_truth_eigenvals[idx[2]])^k for k in 1:length(lambda_iters)]
    # Plot eigenvalue differences
    p = plot(1:length(lambda_iters), eigenvalue_differences, 
    xlabel="Iteration", ylabel="Value", 
    title="Convergence of Differences", 
    label="|lambda(m) - lambda1|", 
    lw=2)
    # Overlay the λ₁/λ₂ values on the same graph
    plot!(1:length(lambda_iters), lambda_1_o_2, 
    label="(lambda1/lambda2)^k", 
    lw=2, linestyle=:dash, color=:red)
    # Overlay the eigenvector differences on the same graph
    plot!(1:size(y_iters, 2), eigenvector_differences, 
    label="‖y(m) - y1‖", 
    lw=2, linestyle=:dot, color=:blue)
    display(p)
end

function ex6(x_sol, Y)
    x_sol_nrmlzd = x_sol/norm(x_sol,2);
    # Plot normalized x_sol_nrmlzd
    p = plot(x_sol_nrmlzd, label="x_sol_nrmlzd", lw=2, xlabel="Index", ylabel="Value", 
    title="Normalized x_sol and 5 Smallest Eigenvectors")
    # Overlay the 5 smallest eigenvectors
    for i in 1:5
        plot!(Y[i,:], label="Eigenvector $i", lw=1, linestyle=:dash)
    end
    display(p)
end

function ex7(n, v, w, L)
    x = randn(n,n)
    Hx = Hmatvec(L,v,w,x)
    L2 = rand_schrodinger_2d(L, v, w, n)
    # Very high condition number: big rounding errors... condition number also increases with grid size
    # println(cond(Array(L))) 
    Hx_gt = L2*vec(x)
    comparison = norm(vec(Hx)-Hx_gt,2) / (n^2)
    return comparison
end

function ex8(K, min_eps)
    N = [50,100,200]
    comparisons = Dict()
    for n in N
        L = Lap_sparse(n)
        v = generate_random_diagonal_matrix(n)
        w = generate_random_diagonal_matrix(n)
        b = ones(n, n)
        x0 = randn(n, n)
        # @printf("Condition number for n=%d: %f \n", n, cond(L))
        x_sol = conjugate_gradient_2D(L, v, w, b, x0, K, min_eps)
        # Generate ground truth
        L2 = rand_schrodinger_2d(L, v, w, n)  # Ground truth L2
        x_gt = L2 \ vec(b)  # Ground truth solution
        # Calculate normalized error
        comparisons[string(n)] = norm(vec(x_sol) - x_gt, 2) / (n^2)
        x_sol_2d = x_sol
        x_gt_2d = reshape(x_gt, n, n)  # Reshape ground truth
        error_field = abs.(x_sol_2d .- x_gt_2d)  # Compute absolute error field
        p1 = heatmap(x_sol_2d, title="Computed Solution (n=$n)", xlabel="x", ylabel="y", colorbar=true, titlefontsize=10)
        p2 = heatmap(error_field, title="Error Field (Condition Number: $(round(cond(L)))) (n=$n)", xlabel="x", ylabel="y", colorbar=true,  titlefontsize=10)
        p = plot(p1, p2, layout=(1, 2), size=(800, 400))
        display(p)
    end 
    return comparisons
end

function ex9(n, v, w, L, min_eps, n_eigen)
    L2 = rand_schrodinger_2d(L, v, w, n)  # Ground truth L2
    ground_truth_eigenvals = sort(eigvals(Matrix(L2)))
    x = randn(n, n)
    Lambda, Y = deflation_min_2D(L, v, w, x, min_eps, n_eigen)
    comparison = norm(Lambda-ground_truth_eigenvals[1:n_eigen], 2)
    return comparison, Y
end

function ex10(n, v, w, L, K, min_eps, Y)
    x_sol = vec(conjugate_gradient_2D(L, v, w, ones(n, n), zeros(n, n), K, min_eps))
    x_sol_nrmlzd = x_sol / norm(x_sol, 2)
    p = plot(x_sol_nrmlzd, label="x_sol_nrmlzd", lw=2, xlabel="Index", ylabel="Value", title="Normalized x_sol and 5 Smallest Eigenvectors")
    for i in 1:5
        plot!(Y[i, :], label="Eigenvector $i", lw=1, linestyle=:dash)
    end
    display(p)
end

function main_1D()
    ##EX1##
    n = 1000
    Lpv = ex1(n)

    ##EX2##
    K = 500
    min_eps = 10e-7
    x_sol, comparison = ex2(Lpv, n, K, min_eps)
    @printf("CG to ground truth comparison: %f \n", comparison)

    ##EX3##
    ground_truth_eigenvals = sort(eigvals(Matrix(Lpv)))
    comparison = ex3(Lpv, ground_truth_eigenvals, min_eps)
    @printf("Smallest eigenvalue to ground truth comparison: %f \n", comparison)

    ##EX4##
    n_eigen = 5
    Lambda, Y, comparison = ex4(Lpv, n, min_eps, n_eigen, ground_truth_eigenvals)
    @printf("5 smallest eigenvalues to ground truth comparison: %f \n", comparison)

    ##EX5##
    ex5(Lpv, rand(n), min_eps)

    ##EX6##
    ex6(x_sol, Y)

end

function main_2D()
    n = 50
    v = generate_random_diagonal_matrix(n)
    w = generate_random_diagonal_matrix(n)
    L = Lap_sparse(n)

    K = 1000
    min_eps = 10e-6

    ##EX7##
    comparison = ex7(n, v, w, L)
    # It may actually be that our implementation is more accurate, 
    # since it does less FLOPs O(n^3) than kron product O(n^4) and thus less rounding errors? 
    @printf("Hx to ground truth comparison: %f \n", comparison)

    ##EX8##
    ex8(K, min_eps)

    ##EX9##
    n_eigen = 5
    comparison, Y = ex9(n, v, w, L, min_eps, n_eigen)
    @printf("5 smallest eigenvalues to ground truth comparison: %f \n", comparison)

    ##EX10##
    ex10(n, v, w, L, K, min_eps, Y)
end

main_1D()
main_2D()
