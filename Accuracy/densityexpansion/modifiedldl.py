import sympy
import symnum.numpy as snp 
from sympy import symbols, simplify 
import densityexpansion.matrixexp as matexp 
import simsde.operators as operators  

def LL_transition_density(drift_func):

    def transition_density(x_t, x_0, θ, t):
        dim_x = x_0.shape[0]
        ε, γ, *_ = θ 
        μ_1, μ_2 = symbols('μ_1, μ_2')
        Σ_RR, Σ_RS, Σ_SS = symbols('Σ_RR, Σ_RS, Σ_SS')
        V_R, V_S, J_SS = symbols('V_R, V_S, J_SS')
        J_sym = snp.array([[-1, 0], [-1/ε, J_SS]])  
        μ_sym = snp.array([μ_1, μ_2])
        Σ_sym = sympy.Matrix(snp.array([[Σ_RR, Σ_RS], [Σ_RS, Σ_SS]]))
        Σ_sym_inv = snp.array(Σ_sym.inverse_CH())
        x_t_minus_μ_sym = x_t - x_0 - μ_sym
        quadratic = x_t_minus_μ_sym.T @ Σ_sym_inv 
        quadratic = quadratic @ x_t_minus_μ_sym

        log_gauss_kernel = - quadratic / 2 - snp.log((2 * snp.pi)**(dim_x) * Σ_sym.det()) / 2
        density = snp.exp(log_gauss_kernel)

        drift_func_sym = snp.array([V_R, V_S])
        μ, Σ = matexp.mean_and_covariance(x_0, θ, t, drift_func_sym, J_sym)
        drift = snp.array(drift_func(x_0, θ))
        J_drift = operators.diff(drift_func(x_0, θ), x_0)
        μ = μ.subs({drift_func_sym[0]:drift[0], drift_func_sym[1]:drift[1], J_sym[1,1]:J_drift[1,1]})
        Σ = Σ.subs({drift_func_sym[0]:drift[0], drift_func_sym[1]:drift[1], J_sym[1,1]:J_drift[1,1]})
        ret = density.subs({μ_sym[0]:μ[0], μ_sym[1]:μ[1], Σ_sym[0,1]:Σ[0,1], Σ_sym[0,0]:Σ[0,0], Σ_sym[1,1]:Σ[1,1], J_sym[1,1]:J_drift[1,1]})
        return ret

    return transition_density 

def Density_expansion_J_3(drift_func):

    def transition_density(x_t, x_0, θ, t):
        dim_x = x_0.shape[0]
        ε, γ, α, σ = θ 
        μ_1, μ_2 = symbols('μ_1, μ_2')
        Σ_RR, Σ_RS, Σ_SS = symbols('Σ_RR, Σ_RS, Σ_SS')
        V_R, V_S, J_SS = symbols('V_R, V_S, J_SS')
        J_sym = snp.array([[-1, 0], [-1/ε, J_SS]])  
        μ_sym = snp.array([μ_1, μ_2])
        Σ_sym = sympy.Matrix(snp.array([[Σ_RR, Σ_RS], [Σ_RS, Σ_SS]]))
        Σ_sym_inv = snp.array(Σ_sym.inverse_CH())
        x_t_minus_μ_sym = x_t - x_0 - μ_sym
        quadratic = x_t_minus_μ_sym.T @ Σ_sym_inv 
        quadratic = quadratic @ x_t_minus_μ_sym

        log_gauss_kernel = - quadratic / 2 - snp.log((2 * snp.pi)**(dim_x) * Σ_sym.det()) / 2 

        expA = snp.array(simplify(sympy.Matrix(t * J_sym).exp()))
        x_t_minus_μ_sym = snp.array(x_t_minus_μ_sym)
        expAT_Σinv  = expA.T @ Σ_sym_inv 
        Hermite_first = sympy.Matrix(expAT_Σinv @ x_t_minus_μ_sym)
        
        s = 0.01

        # coefficients with the size of O (t^{1.5}) 
        e_3 = - t**(1/2) * (s - x_0[1] + x_0[1]**3 + x_0[0]) * γ / (2 * ε) * Hermite_first[0]
        e_3 += - t**(3/2) * (6 * x_0[1] * (s - x_0[1] + x_0[1] ** 3 + x_0[0])**2 + 2*(s - x_0[1] + x_0[1] ** 3 + x_0[0])*γ*ε)/ (6 * ε**3 ) * Hermite_first[1]
 
        weight = t ** (3/2) * e_3
        density = snp.exp(log_gauss_kernel + weight - weight**2 /2)

        drift_func_sym = snp.array([V_R, V_S])
        μ, Σ = matexp.mean_and_covariance(x_0, θ, t, drift_func_sym, J_sym)
        drift = snp.array(drift_func(x_0, θ))
        J_drift = operators.diff(drift_func(x_0, θ), x_0)
        μ = μ.subs({drift_func_sym[0]:drift[0], drift_func_sym[1]:drift[1], J_sym[1,1]:J_drift[1,1]})
        Σ = Σ.subs({drift_func_sym[0]:drift[0], drift_func_sym[1]:drift[1], J_sym[1,1]:J_drift[1,1]})
    
        ret = density.subs({μ_sym[0]:μ[0], μ_sym[1]:μ[1], Σ_sym[0,1]:Σ[0,1], Σ_sym[0,0]:Σ[0,0], Σ_sym[1,1]:Σ[1,1], J_sym[1,1]:J_drift[1,1]})
        return ret

    return transition_density


def Density_expansion_J_4(drift_func):

    def transition_density(x_t, x_0, θ, t):
        dim_x = x_0.shape[0]
        ε, γ, α, σ = θ 
        μ_1, μ_2 = symbols('μ_1, μ_2')
        Σ_RR, Σ_RS, Σ_SS = symbols('Σ_RR, Σ_RS, Σ_SS')
        V_R, V_S, J_SS = symbols('V_R, V_S, J_SS')
        J_sym = snp.array([[-1, 0], [-1/ε, J_SS]])  
        μ_sym = snp.array([μ_1, μ_2])
        Σ_sym = sympy.Matrix(snp.array([[Σ_RR, Σ_RS], [Σ_RS, Σ_SS]]))
        Σ_sym_inv = snp.array(Σ_sym.inverse_CH())
        # display(Σ_sym_inv)
        x_t_minus_μ_sym = x_t - x_0 - μ_sym
        quadratic = x_t_minus_μ_sym.T @ Σ_sym_inv 
        quadratic = quadratic @ x_t_minus_μ_sym

        log_gauss_kernel = - quadratic / 2 - snp.log((2 * snp.pi)**(dim_x) * Σ_sym.det()) / 2 

        expA = snp.array(simplify(sympy.Matrix(t * J_sym).exp()))
        x_t_minus_μ_sym = snp.array(x_t_minus_μ_sym)
        expAT_Σinv  = expA.T @ Σ_sym_inv 
        Hermite_first = sympy.Matrix(expAT_Σinv @ x_t_minus_μ_sym)
        expAT_Σinv_expA = sympy.Matrix(expAT_Σinv @ expA)

        H_RR = Hermite_first[0] * Hermite_first[0] - expAT_Σinv_expA[0,0]
        H_RS = Hermite_first[0] * Hermite_first[1] - expAT_Σinv_expA[0,1]
        H_SS = Hermite_first[1] * Hermite_first[1] - expAT_Σinv_expA[1,1]
        
        s = 0.01

        # coefficients with the size of O (t^{1.5}) 
        e_3 = - t**(1/2) * (s - x_0[1] + x_0[1]**3 + x_0[0]) * γ / (2 * ε) * Hermite_first[0]
        e_3 += - t**(3/2) * (6 * x_0[1] * (s - x_0[1] + x_0[1] ** 3 + x_0[0])**2 + 2*(s - x_0[1] + x_0[1] ** 3 + x_0[0])*γ*ε)/ (6 * ε**3 ) * Hermite_first[1]

        # coefficients with the size of O (t^2)
        e_4 = 0.0
        e_4 = - t * γ * σ**2 / (6 * ε) * H_RR
        e_4 += - t ** 2 * σ**2 * (18 * x_0[1] * (s + x_0[1] ** 3 - x_0[1] + x_0[0]) + 4*γ*ε) / (24 * ε ** 3) * H_RS  
        e_4 += - t ** 3 * σ**2 * (24 * x_0[1] * (s + x_0[1] ** 3 - x_0[1] + x_0[0]) + 4*γ*ε) / (120 * ε ** 4) * H_SS 
        
     
        weight = t ** (3/2) * e_3 + t ** 2 * e_4
        density = snp.exp(log_gauss_kernel + weight - weight**2 /2)

        drift_func_sym = snp.array([V_R, V_S])
        μ, Σ = matexp.mean_and_covariance(x_0, θ, t, drift_func_sym, J_sym)
        drift = snp.array(drift_func(x_0, θ))
        J_drift = operators.diff(drift_func(x_0, θ), x_0)
        μ = μ.subs({drift_func_sym[0]:drift[0], drift_func_sym[1]:drift[1], J_sym[1,1]:J_drift[1,1]})
        Σ = Σ.subs({drift_func_sym[0]:drift[0], drift_func_sym[1]:drift[1], J_sym[1,1]:J_drift[1,1]})
    
        ret = density.subs({μ_sym[0]:μ[0], μ_sym[1]:μ[1], Σ_sym[0,1]:Σ[0,1], Σ_sym[0,0]:Σ[0,0], Σ_sym[1,1]:Σ[1,1], J_sym[1,1]:J_drift[1,1]})
        return ret

    return transition_density


def Density_expansion_J_5(drift_func):

    def transition_density(x_t, x_0, θ, t):
        dim_x = x_0.shape[0]
        ε, γ, α, σ = θ 
        μ_1, μ_2 = symbols('μ_1, μ_2')
        Σ_RR, Σ_RS, Σ_SS = symbols('Σ_RR, Σ_RS, Σ_SS')
        V_R, V_S, J_SS = symbols('V_R, V_S, J_SS')
        J_sym = snp.array([[-1, 0], [-1/ε, J_SS]])  
        μ_sym = snp.array([μ_1, μ_2])
        Σ_sym = sympy.Matrix(snp.array([[Σ_RR, Σ_RS], [Σ_RS, Σ_SS]]))
        Σ_sym_inv = snp.array(Σ_sym.inverse_CH())
        # display(Σ_sym_inv)
        x_t_minus_μ_sym = x_t - x_0 - μ_sym
        quadratic = x_t_minus_μ_sym.T @ Σ_sym_inv 
        quadratic = quadratic @ x_t_minus_μ_sym

        log_gauss_kernel = - quadratic / 2 - snp.log((2 * snp.pi)**(dim_x) * Σ_sym.det()) / 2 

        expA = snp.array(simplify(sympy.Matrix(t * J_sym).exp()))
        x_t_minus_μ_sym = snp.array(x_t_minus_μ_sym)
        expAT_Σinv  = expA.T @ Σ_sym_inv 
        Hermite_first = sympy.Matrix(expAT_Σinv @ x_t_minus_μ_sym)
        expAT_Σinv_expA = sympy.Matrix(expAT_Σinv @ expA)

        H_RR = Hermite_first[0] * Hermite_first[0] - expAT_Σinv_expA[0,0]
        H_RS = Hermite_first[0] * Hermite_first[1] - expAT_Σinv_expA[0,1]
        H_SS = Hermite_first[1] * Hermite_first[1] - expAT_Σinv_expA[1,1]
        H_SRR = Hermite_first[1] * Hermite_first[0] * Hermite_first[0] -  Hermite_first[1] * expAT_Σinv_expA[0,0] - 2 * Hermite_first[0] * expAT_Σinv_expA[1,0]
        H_SSR = Hermite_first[1] * Hermite_first[1] * Hermite_first[0] -  Hermite_first[0] * expAT_Σinv_expA[1,1] - 2 * Hermite_first[1] * expAT_Σinv_expA[1,0]
        H_SSS = Hermite_first[1] * (Hermite_first[1] * Hermite_first[1] - 3 * expAT_Σinv_expA[1,1])

    
        s = 0.01

        # coefficients with the size of O (t^{1.5}) 
        e_3 = - t**(1/2) * (s - x_0[1] + x_0[1]**3 + x_0[0]) * γ / (2 * ε) * Hermite_first[0]
        e_3 += - t**(3/2) * (6 * x_0[1] * (s - x_0[1] + x_0[1] ** 3 + x_0[0])**2 + 2*(s - x_0[1] + x_0[1] ** 3 + x_0[0])*γ*ε)/ (6 * ε**3 ) * Hermite_first[1]

        # coefficients with the size of O (t^2)
        e_4 = - t * γ * σ**2 / (6 * ε) * H_RR
        e_4 += - t ** 2 * σ**2 * (18 * x_0[1] * (s + x_0[1] ** 3 - x_0[1] + x_0[0]) + 4*γ*ε) / (24 * ε ** 3) * H_RS  
        e_4 += - t ** 3 * σ**2 * (24 * x_0[1] * (s + x_0[1] ** 3 - x_0[1] + x_0[0]) + 4*γ*ε) / (120 * ε ** 4) * H_SS 
        
        # coefficients with the size of O (t^{2.5}) 
        term_1 = γ * (3*x_0[1]**2 - 1) * (s + x_0[1]**3 - x_0[1] + x_0[0])
        term_1 += - γ * ε * (α + 2*s + 2 * x_0[1]**3 + (γ - 2)*x_0[1] + x_0[0])  
        
        term_2 = α * γ * ε ** 2 - 2 * (s ** 3) - 6 * (s ** 2) * (x_0[1] ** 3 - x_0[1] + x_0[0])
        term_2 += s * (
            γ * ε ** 2 - 6 * (x_0[1] ** 3 - x_0[1] + x_0[0]) ** 2 
            + 6 * x_0[0] * ε * (α + γ * x_0[1] - x_0[0])
            )
        term_2 += - 2 * x_0[1] ** 9 + 6 * x_0[1] ** 7 - 6 * x_0[1] ** 6 * x_0[0] + 6 * x_0[1] ** 5 * (γ * ε - 1) 
        term_2 += 6 * x_0[1] ** 4 * (α * ε - x_0[0] * (ε - 2)) + x_0[1] ** 3 * (γ * (ε - 6) * ε - 6 * x_0[0] ** 2 + 2)
        term_2 += 6 * x_0[1] ** 2 * (
            x_0[0] * (γ * ε + ε - 1) - α * ε
            )  - 2 * x_0[0] ** 3
        term_2 += x_0[1] * (
            ε * ( (γ - 1) * γ * ε + 2 * σ**2 ) - 6 * x_0[0] ** 2 * (ε - 1) + 6 * α * x_0[0] * ε
            )

        e_5 = t ** (1/2) * term_1 / (6 * ε ** 2) * Hermite_first[0]  
        e_5 += - t ** (3/2) * 3 * term_2 / (24 * ε ** 4) * Hermite_first[1]
        e_5 += - t ** (5/2) * 18 * x_0[1] * σ ** 4 / (120 * ε ** 3) * H_SRR
        e_5 += - t ** (7/2) * 60 * x_0[1] * σ ** 4 / (720 * ε ** 4) * H_SSR
        e_5 += - t ** (9/2) * 60 * x_0[1] * σ ** 4 / (5040 * ε ** 5) * H_SSS 

        weight = t ** (3/2) * e_3 + t ** 2 * e_4 + t ** (5/2) * e_5
        density = snp.exp(log_gauss_kernel + weight - weight**2 /2)
        drift_func_sym = snp.array([V_R, V_S])
        μ, Σ = matexp.mean_and_covariance(x_0, θ, t, drift_func_sym, J_sym)
        drift = snp.array(drift_func(x_0, θ))
        J_drift = operators.diff(drift_func(x_0, θ), x_0)
        μ = μ.subs({drift_func_sym[0]:drift[0], drift_func_sym[1]:drift[1], J_sym[1,1]:J_drift[1,1]})
        Σ = Σ.subs({drift_func_sym[0]:drift[0], drift_func_sym[1]:drift[1], J_sym[1,1]:J_drift[1,1]})
    
        ret = density.subs({μ_sym[0]:μ[0], μ_sym[1]:μ[1], Σ_sym[0,1]:Σ[0,1], Σ_sym[0,0]:Σ[0,0], Σ_sym[1,1]:Σ[1,1], J_sym[1,1]:J_drift[1,1]})
        return ret

    return transition_density