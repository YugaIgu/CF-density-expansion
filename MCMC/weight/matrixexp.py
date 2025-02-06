import sympy
import symnum.numpy as snp 

# dim_x = 2

def P1_func(x, θ, J_sym):
    dim_x = x.shape[0]
    zero_mat = snp.array(sympy.Matrix.zeros(dim_x))
    ident = snp.array(sympy.eye(dim_x))
    P1 = snp.concatenate(
        [
            snp.concatenate([zero_mat, ident], axis=1),
            snp.concatenate([zero_mat, J_sym], axis=1),
        ]
    )
    return P1

def P3_func(x, θ, J_sym):
    dim_x = x.shape[0]

    def diff_matrix_FHN(x, θ):
        ε, γ, β, σ = θ
        return snp.array([[σ**2, 0], [0, 0]])

    zero_mat = snp.array(sympy.Matrix.zeros(dim_x)) 
    a = diff_matrix_FHN(x, θ)
    P3 = snp.concatenate(
        [
            snp.concatenate([J_sym, a], axis=1),
            snp.concatenate([zero_mat, -J_sym.T], axis=1),
        ]
    )
    return P3 

def R0_func(x, θ, t, J_sym):
    dim_x = x.shape[0]
    P1 =  sympy.Matrix(t * P1_func(x, θ, J_sym))
    Mat_exp = P1.exp()
    return Mat_exp[:dim_x, dim_x:] 

def Omega_LL_func(x, θ, t, J_sym):
    dim_x = x.shape[0]
    P3 =  sympy.Matrix(t * P3_func(x, θ, J_sym))
    P3 = sympy.Matrix(P3) 
    P, D = P3.diagonalize()
    Mat_exp = P @ D.exp() @ P.inv()
    B, C = Mat_exp[:dim_x,:dim_x], Mat_exp[:dim_x,dim_x:]
    return C @ B.T

def mean_and_covariance(x, θ, t, drift_func_sym, J_sym):
    m = R0_func(x, θ, t, J_sym) @ drift_func_sym
    Σ = Omega_LL_func(x, θ, t, J_sym)
    return m, Σ