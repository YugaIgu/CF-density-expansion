import sympy
import symnum.numpy as snp 
from sympy import symbols
import weight.matrixexp as matexp
import simsde.operators as operators  

def log_weight_DE_J_3_FHN(drift_func):

    def log_one_step_weight_function(x_t, x_0, θ, t):
        ε, γ, β, σ = θ 
        s = 0.00
        
        μ_1, μ_2 = symbols('μ_1, μ_2')
        Σ_RR, Σ_RS, Σ_SS = symbols('Σ_RR, Σ_RS, Σ_SS')
        V_R, V_S, J_SS = symbols('V_R, V_S, J_SS')
        J_sym = snp.array([[-1, 0], [-1/ε, J_SS]])  
        μ_sym = snp.array([μ_1, μ_2])
        Σ_sym = sympy.Matrix(snp.array([[Σ_RR, Σ_RS], [Σ_RS, Σ_SS]]))
        Σ_sym_inv = snp.array(Σ_sym.inverse_CH())
        x_t_minus_μ_sym = x_t - x_0 - μ_sym
        
        expA = snp.array(sympy.Matrix(t * J_sym).exp())
        x_t_minus_μ_sym = snp.array(x_t_minus_μ_sym)
        expAT_Σinv  = expA.T @ Σ_sym_inv 
        
        # Hermite polynomials 
        Hermite_first = sympy.Matrix(expAT_Σinv @ x_t_minus_μ_sym)        
        
        # coefficients with the size of O (t^{1.5})
        e_3 = - t**(1/2) * (s - x_0[1] + x_0[1]**3 + x_0[0]) * γ / (2 * ε) * Hermite_first[0]
        e_3 += - t**(3/2) * (6 * x_0[1] * (s - x_0[1] + x_0[1] ** 3 + x_0[0])**2 + 2*(s - x_0[1] + x_0[1] ** 3 + x_0[0])*γ*ε)/ (6 * ε**3) * Hermite_first[1]
         
        weight = t ** (3/2) * e_3
        ret = weight - weight**2 /2
        
        # substitution with the true drift and diffusion function 
        drift_func_sym = snp.array([V_R, V_S])
        μ, Σ = matexp.mean_and_covariance(x_0, θ, t, drift_func_sym, J_sym)
        drift = snp.array(drift_func(x_0, θ))
        J_drift = operators.diff(drift_func(x_0, θ), x_0)

        μ = μ.subs(
            {
                drift_func_sym[0]:drift[0], 
                drift_func_sym[1]:drift[1], 
                J_sym[1,1]:J_drift[1,1]
            })

        Σ = Σ.subs(
            {
                drift_func_sym[0]:drift[0], 
                drift_func_sym[1]:drift[1], 
                J_sym[1,1]:J_drift[1,1]
            }
            )
        
        ret = ret.subs(
            {
                μ_sym[0]:μ[0], 
                μ_sym[1]:μ[1], 
                Σ_sym[0,1]:Σ[0,1], 
                Σ_sym[0,0]:Σ[0,0], 
                Σ_sym[1,1]:Σ[1,1], 
                J_sym[1,1]:J_drift[1,1]
            }
            )
        
        return ret

    return log_one_step_weight_function

def log_weight_DE_J_4_FHN(drift_func):

    def log_one_step_weight_function(x_t, x_0, θ, t):
        ε, γ, β, σ = θ
        s = 0.00
        
        μ_1, μ_2 = symbols('μ_1, μ_2')
        Σ_RR, Σ_RS, Σ_SS = symbols('Σ_RR, Σ_RS, Σ_SS')
        V_R, V_S, J_SS = symbols('V_R, V_S, J_SS')
        J_sym = snp.array([[-1, 0], [-1/ε, J_SS]])  
        μ_sym = snp.array([μ_1, μ_2])
        Σ_sym = sympy.Matrix(snp.array([[Σ_RR, Σ_RS], [Σ_RS, Σ_SS]]))
        Σ_sym_inv = snp.array(Σ_sym.inverse_CH())
        x_t_minus_μ_sym = x_t - x_0 - μ_sym
        # print(log_gauss_kernel)

    
        # expA = snp.array(simplify(sympy.Matrix(t * J_sym).exp()))
        expA = snp.array(sympy.Matrix(t * J_sym).exp())
        x_t_minus_μ_sym = snp.array(x_t_minus_μ_sym)
        expAT_Σinv  = expA.T @ Σ_sym_inv 
        expAT_Σinv_expA = sympy.Matrix(expAT_Σinv @ expA)

        # Hermite polynomials 
        Hermite_first = sympy.Matrix(expAT_Σinv @ x_t_minus_μ_sym)        
        H_RR = Hermite_first[0] * Hermite_first[0] - expAT_Σinv_expA[0,0]
        H_RS = Hermite_first[0] * Hermite_first[1] - expAT_Σinv_expA[0,1]
        H_SS = Hermite_first[1] * Hermite_first[1] - expAT_Σinv_expA[1,1]
        
        # coefficients with the size of O (t^{1.5})
        e_3 = - t**(1/2) * (s - x_0[1] + x_0[1]**3 + x_0[0]) * γ / (2 * ε) * Hermite_first[0]
        e_3 += - t**(3/2) * (6 * x_0[1] * (s - x_0[1] + x_0[1] ** 3 + x_0[0])**2 + 2*(s - x_0[1] + x_0[1] ** 3 + x_0[0])*γ*ε)/ (6 * ε**3) * Hermite_first[1]

        # coefficients with the size of O (t^2)
        e_4 = 0.0
        e_4 = - t * γ * σ**2 / (6 * ε) * H_RR
        e_4 += - t ** 2 * σ**2 * (18 * x_0[1] * (s + x_0[1] ** 3 - x_0[1] + x_0[0]) + 4*γ*ε) / (24 * ε ** 3) * H_RS  
        e_4 += - t ** 3 * σ**2 * (24 * x_0[1] * (s + x_0[1] ** 3 - x_0[1] + x_0[0]) + 4*γ*ε) / (120 * ε ** 4) * H_SS 
        
        weight = t ** (3/2) * e_3 + t ** 2 * e_4
        ret = weight - weight**2 /2
        
        # substitution with the true drift and diffusion function 
        drift_func_sym = snp.array([V_R, V_S])
        μ, Σ = matexp.mean_and_covariance(x_0, θ, t, drift_func_sym, J_sym)
        drift = snp.array(drift_func(x_0, θ))
        J_drift = operators.diff(drift_func(x_0, θ), x_0)

        μ = μ.subs(
            {
                drift_func_sym[0]:drift[0], 
                drift_func_sym[1]:drift[1], 
                J_sym[1,1]:J_drift[1,1]
            })

        Σ = Σ.subs(
            {
                drift_func_sym[0]:drift[0], 
                drift_func_sym[1]:drift[1], 
                J_sym[1,1]:J_drift[1,1]
            }
            )
        
        ret = ret.subs(
            {
                μ_sym[0]:μ[0], 
                μ_sym[1]:μ[1], 
                Σ_sym[0,1]:Σ[0,1], 
                Σ_sym[0,0]:Σ[0,0], 
                Σ_sym[1,1]:Σ[1,1], 
                J_sym[1,1]:J_drift[1,1]
            }
            )
        
        return ret

    return log_one_step_weight_function
