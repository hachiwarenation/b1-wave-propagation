import numpy as np

class Solver:
    """Class for solving a 1D 1st-order PDE, the advection equation
    Discretisation methods are implemented as methods"""
    def __init__(self,c:float,N:int,M:int,t_F:float=1,x_F:float=3):
        # Not sure on the best way to pass the arguments
        self.c = c
        self.N = N
        self.M = M
        self.t_F = t_F
        self.x_F = x_F
        # Handy values
        self.dt = t_F/(N-1)
        self.dx = x_F/(M-1)

    def FTCS(self,u_0:np.ndarray,u_x_F=[0,0]):
        """u_0 is the initial condition as an array size 1,2M-1
        u_x_F is a tuple specifying the value of u to populate at each of
        [-x_F,t] and [x_F,t] for all time
        Output an array size N,2M-1 where the nth row corresponds to the
        system after n*dt time steps"""
        u = np.zeros((self.N,2*self.M-1))
        u[0] = u_0
        for n in range(self.N-1):
            u[n+1][0] = u_x_F[0]
            u[n+1][-1] = u_x_F[1]
            for m in range(1,2*self.M-2):
                u[n+1][m] = -self.c*self.dt*(u[n][m+1]-u[n][m-1])/2/self.dx + u[n][m]
        return u
    
    def get_x(self):
        return np.linspace(-self.x_F,self.x_F,2*self.M-1)
    
    def get_t(self):
        return np.linspace(0,self.t_F,self.N)
    
    def get_courant(self):
        return self.c*self.dt/self.dx


# Test waves
def wave_f(x):
    return np.piecewise(x, [abs(x)<1, abs(x)>=1],
                        [lambda y: np.exp(-1/(1-y**2)), 0])

def wave_g(x):
    return np.piecewise(x, [x<=0, (x>0)&(x<1), x>=1],
                    [0, lambda y: y, 0])


# Run a test
def run_test(solver,method,test_func):
    return getattr(solver,method)(test_func(solver.get_x()))




