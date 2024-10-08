import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

### Plotting routines
def plot_chain(z, a, b, legend = [], save_fig = False):
    N = int(len(z) / 2)
    xc = np.zeros(N+2)
    yc = np.zeros(N+2)
    xc[1:N+1] = z[0:N]
    xc[N+1] = a
    yc[1:N + 1] = z[N:2*N]
    yc[N + 1] = b
    plt.figure("Chain")
    x = np.linspace(0, 1, 30)
    #plt.plot(x,-0.2*x-0.1, label='Contrainte')   # affichage de la contrainte
    plt.plot(xc,yc, 'ko-',label='Chaine')
    plt.grid()
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    return


### Elements to compute the Newton's method

def cost(z):
    N = int(len(z)/2)
    e = np.zeros(2*N)
    e[N:2*N] = np.ones(N)
    return np.dot(e, z)

def constraint_value(z, L, a, b):
    N = int(len(z) / 2)
    # code the constraint c
    c = np.zeros(N+1)
    c[0] = (z[0] - 0.) ** 2 + (z[N] - 0.) ** 2 - L ** 2
    for i in range(1,N):
        c[i] = (z[i] - z[i-1])**2 + (z[N+i]-z[N+i - 1])**2 - L**2
    c[N] = (a - z[N-1]) ** 2 + (b-z[2*N-1]) ** 2 - L ** 2
    return c

def nabla_res(z, L, a, b):

    N = int(len(z) / 2)
    C = np.zeros([2*N, N+1])
    for i in range(1,N):
        C[i,i] = 2*(z[i]-z[i-1])
        C[i+N,i] = 2*(z[i+N]-z[i-1+N])
        if i<N-1:
            C[i,i+1] = -2*(z[i+1]-z[i])
            C[i+N,i+1] = -2*(z[i+1+N]-z[i+N])

    C[0,0] = 2*(z[0]-0.)
    C[N,0] = 2*(z[N]-0.)
    C[0, 1] = 2 * (z[0] - z[1])
    C[N, 1] = 2 * (z[N] - z[N+1])
    C[N-1,N] = -2*(a-z[N-1])
    C[2*N-1,N] = -2*(b-z[2*N-1])

    return np.mat(C)

def check_der(z, L, a, b):
    N = int(len(z) / 2)
    dz = 0.01*np.random.random(2*N)
    C = nabla_res(z, L, a, b)
    der = (constraint_value(z+dz, L, a, b) - constraint_value(z, L, a, b)) - np.dot(nabla_res(z, L, a, b).transpose(), dz)
    print('NORM ERROR: ', np.linalg.norm(der)/np.linalg.norm(dz))

    return


def nabla_F(z, lbd):
    """
    nabla_F computes the upper left block of the Newton's method Hessian
    """
    N = int(len(z) / 2)
    nF = np.zeros([2*N, 2*N])
    for i in range(N):
        if i>0:
            nF[i,i-1] = -2*lbd[i]
            nF[N + i, N + i - 1] = -2 * lbd[i]

        nF[i, i] = 2 * (lbd[i]+lbd[i+1])
        nF[N + i, N + i] = 2 * (lbd[i] + lbd[i + 1])


        if i<N-1:
            nF[i, i+1] = -2 * lbd[i+1]
            nF[N+i, N+i + 1] = -2 * lbd[i + 1]

    return nF


def newton_iteration_elements(z, lbd, L, a, b):
    N = int(len(z) / 2)
    # build system

    A11 = nabla_F(z, lbd)
    A12 = nabla_res(z, L, a, b)
    A21 = A12.transpose()
    A22 = np.zeros([N + 1, N + 1])

    A1r = np.concatenate((A11, A12), axis=1)
    A2r = np.concatenate((A21, A22), axis=1)
    A = np.concatenate((A1r, A2r), axis=0)
    #
    e = np.zeros(2 * N)
    e[N:2 * N] = np.ones(N)
    rhs1 = e + np.dot(A12, lbd)
    rhs2 = constraint_value(z, L, a, b)
    rhs = np.zeros([3 * N + 1])
    rhs[:2 * N] = rhs1
    rhs[2 * N:] = rhs2

    return A, rhs

def newton_iteration(z, lbd, L, a, b, backtracking=True):
    N = int(len(z) / 2)
    # build system
    A, rhs = newton_iteration_elements(z, lbd, L, a, b)
    # solve system
    d = np.linalg.solve(A, -rhs)
    dz=d[:2*N]
    dlbd=d[2*N:]
    if (backtracking==True):
        alp=1
        tau=0.5
        beta=0.25
        A1, rhs1 = newton_iteration_elements(z+alp*dz,lbd+alp*dlbd,L,a,b)
        while (np.linalg.norm(rhs1)>(1-beta*alp)*np.linalg.norm(rhs)):
            alp=tau*alp
            A1, rhs1 = newton_iteration_elements(z+alp*dz,lbd+alp*dlbd,L,a,b)
        z=z+alp*dz
        lbd=lbd+alp*dlbd
    else:
        z = z + d[0:2*N]
        lbd = lbd + d[2*N:3*N+1] 
    gap = np.linalg.norm(rhs)
    return z, lbd, gap



def solve_chain(z0, lbd0, L, a, b, Nmax=100, tol=1e-6, backtracking=True):
    """
    this solves the chain with your Newton method. This controls the number of iterations and
    the stopping criterion
    """
    err, k = 1, 0
    z= z0
    lbd = lbd0
    g = list()
    while (k < Nmax) and (err > tol):
        k += 1

        z, lbd, gap = newton_iteration(z, lbd, L, a, b, backtracking=backtracking)

        g.append(gap)
        err = gap

    return z, lbd, g


def check_stationarity(z, lbd, gap):
    if gap[-1] < 1e-3:
        F = nabla_F(z, lbd)
        e, dummy = np.linalg.eigh(F)
        print('Eigenvalues : ', np.min(e), np.max(e))
        if np.min(e) > 0:
            print('A local optimizer found')
    return


def plot_convergence(opt_gap):
    plt.figure("Convergence")
    plt.loglog(opt_gap, 'o-')
    plt.grid()
    plt.xlabel('Iteration count')
    plt.ylabel('Optimality gap (norm residual)')
    plt.show()
    return



N = 10
L = 0.2
a,b = 1.,1.
z = np.zeros(2*N)
z[0:N] = np.linspace(1./(N+1), a-1./(N+1), N)
z[N:2*N] =[b*i -.1*(i-a/2.)**2 for i in z[0:N]]
lmbda = 0.01*np.random.random(N+1)

z_sol, lmbda_sol, opt_gap = solve_chain(z, lmbda, L, a, b, Nmax = 100, tol = 1e-6, backtracking = True)
check_stationarity(z_sol, lmbda_sol, opt_gap)
plot_chain(z_sol, a, b)
plot_convergence(opt_gap)

## Going beyond : Inequalities constraint

#fonction à minimiser
def objective_function(z):
    N = int(len(z)/2)
    S = 0
    for k in range(N,2*N):
        S+=z[k]
    return S

# Fonction pour les contraintes d'égalité basée sur 'constraint_value'
def constraint_eq(z,L,a,b):
    # Utilisation de la fonction 'constraint_value' pour définir les contraintes
    c_values = constraint_value(z, L, a, b)
    return c_values

def constraint_g(z,L,a,b):
    N = int(len(z) / 2)
    # code the constraint g
    g = np.zeros(N+1)
    g[0] = 0.2*z[0] + z[N] + 0.1
    for i in range(1,N):
        g[i] = 0.2*z[i] + z[N+i] + 0.1
    g[N] = 0.2*a + b + 0.1
    return g

cons = (
    {'type': 'eq','fun': lambda z: constraint_eq(z, L, a, b)},
    {'type':'ineq','fun':lambda z: constraint_g(z, L, a, b)}
    )



z0 = np.zeros(2*N)  # Initialistion

# Résolution du problème d'optimisation avec SLSQP
result = minimize(objective_function, z0, method='SLSQP', constraints=cons)

# Affichage des résultats
print("Résultat de l'optimisation :")
print("Succès :", result.success)
print("Statut :", result.message)
print("Valeur optimale :", result.fun)
print("z optimal :", result.x)
plot_chain(result.x,a,b)

