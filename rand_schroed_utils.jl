
using SparseArrays
using LinearAlgebra
using Kronecker
using Printf

"""
Returns a sparse symmetric tridiagonal matrix approximating the Laplacian operator. 

# Arguments
- `n::Int`: The size of the matrix (n × n).

# Returns
- `n×n SymTridiagonal{Int64, Vector{Int64}}`: sparse symmetric tridiagonal matrix approximating the Laplacian operator.
"""
function Lap_sparse(n)
    return SymTridiagonal(2*ones(n), -1*ones(n-1));
end

"""
Generate a random sparse diagonal matrix with diag(M)=1+eps, eps=1/(N^2), with probability 1/2 for each entry.

# Arguments
- `n::Int`: The size of the matrix (n × n).

# Returns
- `SparseMatrixCSC`: The sparse diagonal matrix.
"""
function generate_random_diagonal_matrix(n)
    random_values = rand(Bool, n)
    diagonals = [val ? 1.0 : 1 / n^2 for val in random_values]
    return spdiagm(0 => diagonals)
end


function rand_schrodinger_1d(n)
    Lap_sparse(n)+generate_random_diagonal_matrix(n);
end


function Lap2(L, n)
    """
    Compute the 2D Laplacian matrix L(2) = L krnck I + I krnck L for ground truth comparison.
    Arguments:
    - L: 1D finite difference Laplacian matrix.
    - n: Dimension of the 2D system (grid size NxN).
    Returns:
    - L2: The full 2D Laplacian matrix.
    """
    Id = sparse(I,n,n)
    L2 = kron(Id, L) + kron(L, Id)  
    return L2
end

function rand_schrodinger_2d(L, v, w, n)
    """
    Constructs the full matrix L(2) + v krnck w just for ground truth checks (not 
    used in practice as it is inneficient).
    Arguments:
    - L: 1D finite difference Laplacian matrix.
    - v: 1D potential vector v.
    - w: 1D potential vector w.
    - n: Grid dimension for 2D system.
    Returns:
    - L(2) + v krnck w
    """
    return Lap2(L, n) + kron(v, w)
end


function Hmatvec(L,v,w,x)
    # Compute the 2D Laplacian matrix-vector product: L(2)x = Lx + xL 
    L2x = L * x  + x * L        
    # Compute the element-wise potential term: Vx = (v krnck w) .* x
    Vx = diag(v) .* x .* diag(w)'
    # Combine the two parts: Hx = L(2)x + Vx
    return L2x + Vx
end

function print_comparison(x_sol, x_gt)
    # Reshape flattened vectors back to 2D matrices
    x_sol_2d = reshape(vec(x_sol), size(x_gt))
    x_gt_2d = reshape(x_gt, size(x_gt))

    println("Ground Truth Solution vs Computed Solution:")
    println("-----------------------------------------")
    for i in 1:size(x_gt_2d, 1)
        for j in 1:size(x_gt_2d, 2)
            # Print the comparison row by row
            @printf("GT[%d,%d]: %8.4f   Computed[%d,%d]: %8.4f\n",
                    i, j, x_gt_2d[i, j], i, j, x_sol_2d[i, j])
        end
    end
    println("-----------------------------------------")
    # Print the norm of the difference
    diff_norm = norm(vec(x_sol) - x_gt, 2)
    @printf("Norm of difference: %.4e\n", diff_norm)
end