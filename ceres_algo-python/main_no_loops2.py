from qr_decomposition import *
import numpy as np
import scipy.linalg
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

def levenberg_marquardt(r, J, x, Delta = 100, Delta_max = 10000,
                        eta = 0.0001, sigma = 0.1,
                        nmax = 500, tol_abs = 10**(-7),
                        tol_rel = 10**(-7), eps = 10**(-3),
                        Scaling = False):
    def f(x):
        return 0.5*np.linalg.norm(r(x))**2
    def gradf(x):
        return np.dot(np.transpose(J(x)),r(x))
    counter = 0
    fx = f(x)
    func_eval = 1
    m,n = J(x).shape
    
    Information = [['counter', 'norm of step p',
                    'x', 'norm of the gradient at x']]
    
    Information.append([counter, 'not available', x,
                        np.linalg.norm(gradf(x))])#(x)
    
    tolerance = min((tol_rel * Information[-1][-1] + tol_abs), eps)
    
    D = np.eye(n)
    D_inv = np.eye(n)
    
    while Information[-1][-1] > tolerance and counter < nmax:
        Jx = J(x)
        if Scaling == True:
            # for i in list(range(0, n, 1)):
            #     D[i,i] = max(D[i,i], np.linalg.norm(Jx[:,i]))
            # #объединить
            # for i in list(range(0, n, 1)):
            #     D_inv[i,i] = 1/D[i,i]
            D = np.eye(n) * np.amax(np.abs(Jx), axis=0)
            D_inv = np.linalg.inv(D)

        D_2 = np.dot(D, D)
        Q, R, Pi = scipy.linalg.qr(Jx, pivoting=True)
        P = np.eye(n)[:,Pi]
        rank = np.linalg.matrix_rank(Jx)
        if rank == n:
            p = np.dot(P, scipy.linalg.solve_triangular(
                R[0:n,:], np.dot(Q[:,0:n].T, -r(x))))
        else:
            y = np.zeros(n)
            y[0:rank] = scipy.linalg.solve_triangular(
                R[0:rank,0:rank], np.dot(Q[:,0:rank].T, -r(x)))
            p = np.dot(P, y)
        
        Dp = np.linalg.norm(np.dot(D, p))
        # print('Jx',Jx)
        print('rank',rank) 
        print('n',n)
        #print('Q',Q)  
        # print('Pi',Pi)    
        print('P',P)
        # print('rank',rank)
        # print('XXX',x)
        print('p',p)
        # print('RRR',R)
        print('y',y)
        return
        
        if Dp <= ((1+sigma) * Delta):
            alpha = 0
        else:
            J_scaled = np.dot(Jx, D_inv)
            u = np.linalg.norm(np.dot(J_scaled.T, r(x))) / Delta
            if rank == n:
                q = scipy.linalg.solve_triangular(
                    R[0:n,:].T, np.dot(P.T, np.dot(D_2, p)),
                    lower = True)
                l = (Dp - Delta) / (np.linalg.norm(q)**2 / Dp)
            else:
                l = 0
            
            if u == np.inf:
                alpha = 1
            else:
                alpha = max(0.001 * u, (l * u)**(0.5))
            
            while Dp > (1 + sigma) * Delta or Dp < (1 - sigma) * Delta:
                if alpha == np.inf:
                    print('Error: '
                          + 'The LM method fails to converge.'
                          + '(Lambda gets too large)'
                          + 'Please try a different starting point.')
                    return x, Information
                if alpha <= l or alpha > u:
                    alpha = max(0.001 * u, (l * u)**(0.5))
                
                D_lambda = np.dot(P.T, np.dot(D, P))
                R_I = np.concatenate((R, alpha**(0.5) * D_lambda), axis = 0)
                
                #decomposition usage
                R_lambda, Q_lambda2 = givens_qr(R_I, n, m)
                
                Q_lambda = np.dot(np.concatenate(
                    (np.concatenate((Q, np.zeros((m,n))), axis = 1),
                     np.concatenate((np.zeros((n,m)), P), axis = 1)),
                    axis = 0), Q_lambda2)
                
                r_0 = np.append(r(x), np.zeros(n))
                
                p = np.dot(P, scipy.linalg.solve_triangular(
                    R_lambda[0:n,:], np.dot(Q_lambda[:,0:n].T, -r_0)))
                
                Dp = np.linalg.norm(np.dot(D, p))
                
                q = scipy.linalg.solve_triangular(
                    R_lambda[0:n,:].T,
                    np.dot(P.T, np.dot(D_2, p)), lower = True)
                
                phi = Dp - Delta
                phi_derivative = -np.linalg.norm(q)**2 / Dp
                
                if phi < 0:
                    u = alpha
                l = max(l, alpha - phi / phi_derivative)
                alpha = alpha - ((phi + Delta) / Delta) * (phi / phi_derivative)
        
        fxp = f(x + p) 
        func_eval += 1
        if fxp > fx or fxp == np.inf or np.isnan(fxp) == True:
            rho = 0
        else:
            ared = 1 - (fxp / fx)
            pred = (0.5 * np.linalg.norm(np.dot(Jx, p))**2) / fx + (alpha * Dp**2) / fx
            rho = ared / pred
        
        if rho < 0.25:
            Delta = 0.25 * Delta
        else:
            if rho > 0.75 and Dp >= (1 - sigma) * Delta:
                Delta = min(2 * Delta, Delta_max)
            # else:
            #     Delta = Delta
        if rho > eta:
            x += p
            fx = fxp
            counter += 1
            Information.append([counter, np.linalg.norm(p), x, 
                                np.linalg.norm(gradf(x))])
    
    if Information[-1][-1] <= tolerance:
        print('The LM method terminated successfully.')
        print('\n Current function value: ' + str(fx))
        print('Iterations: ' + str(counter))
        print('Function evaluations: '+ str(func_eval))
    else:
        print('The LM method fails to converge within'+ str(nmax) + 'steps.')
    return x, Information

def r_old(x):
    r = np.zeros((2,))
    r[0] = x[0]**2 + x[1]**2 - 1
    r[1] = x[0] - x[1]**2
    return r

def J_old(x):
    J = np.zeros((2, 2))
    J[0, 0] = 2*x[0]
    J[0, 1] = 2*x[1]
    J[1, 0] = 1
    J[1, 1] = -2*x[1]
    return J

def r_old_2(x):
    r = np.zeros((2,))
    rvec, tvec, camera, xy_obs = x
    p0 = rvec[0] + tvec[0]
    p1 = rvec[1] + tvec[1]
    p2 = rvec[2] + tvec[2]
    xp = -p0/p2
    yp = -p1/p2
    r2 = xp**2 + yp**2
    l1 = 1
    l2 = 1
    distortion = 1.0 + r2*(l1 + l2*r2)
    focal = camera[0][0]
    predicted_x = focal*distortion*xp
    predicted_y = focal*distortion*yp
    
    r[0] = predicted_x - xy_obs[0]
    r[1] = predicted_y - xy_obs[1]
    
    return r

def J_old_2(x):
    
    rvec, tvec, camera, _ = x
    l1 = 1
    l2 = 1
    #l1,l2 берем константой
    focal = camera[0][0]
    J = np.zeros((2, 7))
    
    # dfocal pred_x
    J[0, 0] = -(rvec[0] + tvec[0])*((rvec[2] + tvec[2])**(-1)) - l1*((rvec[0] + tvec[0])**3)*((rvec[2] + tvec[2])**(-3)) - l1*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-3)) - l2*((rvec[0] + tvec[0])**5)*((rvec[2] + tvec[2])**(-5)) - 2*l2*((rvec[0] + tvec[0])**3)*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-5)) - l2*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**4)*((rvec[2] + tvec[2])**(-5))
    # drvec[0] pred_x
    J[0, 1] = -focal*(1)*((rvec[2] + tvec[2])**(-1)) - 3*l1*focal*((rvec[0] + tvec[0])**2)*((rvec[2] + tvec[2])**(-3)) - l1*focal*(1)*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-3)) - 5*l2*focal*((rvec[0] + tvec[0])**4)*((rvec[2] + tvec[2])**(-5)) - 6*l2*focal*((rvec[0] + tvec[0])**2)*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-5)) - l2*focal*(1)*((rvec[1] + tvec[1])**4)*((rvec[2] + tvec[2])**(-5))
    # drvec[1] pred_x
    J[0, 2] = -2*l1*focal*(rvec[0] + tvec[0])*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-3)) - 4*l2*focal*((rvec[0] + tvec[0])**3)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-5)) - 4*l2*focal*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-5))
    # drvec[2] pred_x
    J[0, 3] = focal*(rvec[0] + tvec[0])*(1/(rvec[2] + tvec[2])**(2)) + 3*l1*focal*((rvec[0] + tvec[0])**3)*((rvec[2] + tvec[2])**(-4)) + 3*l1*focal*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-4)) + 5*l2*focal*((rvec[0] + tvec[0])**5)*((rvec[2] + tvec[2])**(-6)) + 10*l2*focal*((rvec[0] + tvec[0])**3)*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-6)) + 5*l2*focal*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**4)*((rvec[2] + tvec[2])**(-6))
    # dtvec[0] pred x
    J[0, 4] = -focal*(1)*((rvec[2] + tvec[2])**(-1)) - 3*l1*focal*((rvec[0] + tvec[0])**2)*((rvec[2] + tvec[2])**(-3)) - l1*focal*(1)*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-3)) - 5*l2*focal*((rvec[0] + tvec[0])**4)*((rvec[2] + tvec[2])**(-5)) - 6*l2*focal*((rvec[0] + tvec[0])**2)*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-5)) - l2*focal*(1)*((rvec[1] + tvec[1])**4)*((rvec[2] + tvec[2])**(-5))
    # dtvec[1] pred x
    J[0, 5] = -2*l1*focal*(rvec[0] + tvec[0])*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-3)) - 4*l2*focal*((rvec[0] + tvec[0])**3)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-5)) - 4*l2*focal*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-5))
    # dtvec[2] pred x
    J[0, 6] = focal*(rvec[0] + tvec[0])*(1/(rvec[2] + tvec[2])**(2)) + 3*l1*focal*((rvec[0] + tvec[0])**3)*((rvec[2] + tvec[2])**(-4)) + 3*l1*focal*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-4)) + 5*l2*focal*((rvec[0] + tvec[0])**5)*((rvec[2] + tvec[2])**(-6)) + 10*l2*focal*((rvec[0] + tvec[0])**3)*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-6)) + 5*l2*focal*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**4)*((rvec[2] + tvec[2])**(-6))
    
    # dfocal pred_y
    J[1, 0] = -(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-1)) - l1*((rvec[0] + tvec[0])**2)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-3)) - l1*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-3)) - l2*((rvec[0] + tvec[0])**4)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-5)) - 2*l2*((rvec[0] + tvec[0])**2)*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-5)) - l2*((rvec[1] + tvec[1])**5)*((rvec[2] + tvec[2])**(-5))
    # drvec[0] pred_y
    J[1, 1] = -2*l1*focal*(rvec[0] + tvec[0])*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-3)) - 4*l2*focal*((rvec[0] + tvec[0])**3)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-5)) - 4*l2*focal*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-5))
    # drvec[1] pred_y
    J[1, 2] = -focal*(1)*((rvec[2] + tvec[2])**(-1)) - l1*focal*((rvec[0] + tvec[0])**2)*(1)*((rvec[2] + tvec[2])**(-3)) - 3*l1*focal*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-3)) - l2*focal*((rvec[0] + tvec[0])**4)*(1)*((rvec[2] + tvec[2])**(-5)) - 6*l2*focal*((rvec[0] + tvec[0])**2)*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-5)) - 5*l2*focal*((rvec[1] + tvec[1])**4)*((rvec[2] + tvec[2])**(-5))
    # drvec[2] pred_y
    J[1, 3] = focal*(rvec[1] + tvec[1])*(1/(rvec[2] + tvec[2])**(2)) + 3*l1*focal*((rvec[0] + tvec[0])**2)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-4)) + 3*l1*focal*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-4)) + 5*l2*focal*((rvec[0] + tvec[0])**4)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-6)) + 10*l2*focal*((rvec[0] + tvec[0])**2)*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-6)) + 5*l2*focal*((rvec[1] + tvec[1])**5)*((rvec[2] + tvec[2])**(-6))
    # dtvec[0] pred_y
    J[1, 4] = -2*l1*focal*(rvec[0] + tvec[0])*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-3)) - 4*l2*focal*((rvec[0] + tvec[0])**3)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-5)) - 4*l2*focal*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-5))
    # dtvec[1] pred_y
    J[1, 5] = -focal*(1)*((rvec[2] + tvec[2])**(-1)) - l1*focal*((rvec[0] + tvec[0])**2)*(1)*((rvec[2] + tvec[2])**(-3)) - 3*l1*focal*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-3)) - l2*focal*((rvec[0] + tvec[0])**4)*(1)*((rvec[2] + tvec[2])**(-5)) - 6*l2*focal*((rvec[0] + tvec[0])**2)*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-5)) - 5*l2*focal*((rvec[1] + tvec[1])**4)*((rvec[2] + tvec[2])**(-5))
    # dtvec[2] pred_y
    J[1, 6] = focal*(rvec[1] + tvec[1])*(1/(rvec[2] + tvec[2])**(2)) + 3*l1*focal*((rvec[0] + tvec[0])**2)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-4)) + 3*l1*focal*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-4)) + 5*l2*focal*((rvec[0] + tvec[0])**4)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-6)) + 10*l2*focal*((rvec[0] + tvec[0])**2)*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-6)) + 5*l2*focal*((rvec[1] + tvec[1])**5)*((rvec[2] + tvec[2])**(-6))
    
    return J

    # Расчеты матрицы Якоби выполнялись вследствие следующих упрощений
    # distortion = 1.0 + l1*(xp**2 + yp**2) + l2*(xp**2 + yp**2)**2
    # predicted_x = focal*xp + focal*xp*l1*(xp**2 + yp**2) + focal*xp*l2*(xp**2 + yp**2)**2
    # predicted_y = focal*yp + focal*yp*l1*(xp**2 + yp**2) + focal*yp*l2*(xp**2 + yp**2)**2
    # predicted_x = focal*(xp) + focal*l1*(xp**3) + focal*l1*(xp)*(yp**2) + focal*l2*(xp**5) + 2*focal*l2*(xp**3)*(yp**2) + focal*l2*(xp)*(yp**4)
    # predicted_y = focal*(yp) + focal*(yp)*l1*(xp**2) + focal*l1*(yp**3) + focal*(yp)*l2*(xp**4) + 2*focal*l2*(xp**2)*(yp**3) + focal*l2*(yp**5)
    # predicted_x = -focal*p0*p2**(-1) - focal*l1*p0**3*p2**(-3) - focal*l1*p0*p2**(-1)*p1**2*p2**(-2) - focal*l2*p0**5*p2**(-5) - 2*focal*l2*p0**3*p2**(-3)*p1**2*p2**(-2) - focal*l2*p0*p2**(-1)*p1**4*p2**(-4)
    # predicted_y = -focal*p1*p2**(-1) - focal*p1*p2**(-1)*l1*p0**2*p2**(-2) - focal*l1*p1**3*p2**(-3) - focal*p1*p2(-1)*l2*p0**4*p2**(-4) - 2*focal*l2*p0**2*p2**(-2)*p1**3*p2**(-3) - focal*l2*p1**5*p2**(-5)
    # predicted_x = -focal*p0*(p2**(-1)) - l1*focal*(p0**3)*(p2**(-3)) - l1*focal*p0*(p1**2)*(p2**(-3)) - l2*focal*(p0**5)*(p2**(-5)) - 2*l2*focal*(p0**3)*(p1**2)*(p2**(-5)) - l2*focal*p0*(p1**4)*(p2**(-5))
    # predicted_y = -focal*p1*(p2**(-1)) - l1*focal*(p0**2)*p1*(p2**(-3)) - l1*focal*(p1**3)*(p2**(-3)) - l2*focal*(p0**4)*p1*(p2**(-5)) - 2*l2*focal*(p0**2)*(p1**3)*(p2**(-5)) - l2*focal*(p1**5)*(p2**(-5))
    # predicted_x = -focal*(rvec[0] + tvec[0])*((rvec[2] + tvec[2])**(-1)) - l1*focal*((rvec[0] + tvec[0])**3)*((rvec[2] + tvec[2])**(-3)) - l1*focal*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-3)) - l2*focal*((rvec[0] + tvec[0])**5)*((rvec[2] + tvec[2])**(-5)) - 2*l2*focal*((rvec[0] + tvec[0])**3)*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-5)) - l2*focal*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**4)*((rvec[2] + tvec[2])**(-5))
    # predicted_y = -focal*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-1)) - l1*focal*((rvec[0] + tvec[0])**2)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-3)) - l1*focal*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-3)) - l2*focal*((rvec[0] + tvec[0])**4)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-5)) - 2*l2*focal*((rvec[0] + tvec[0])**2)*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-5)) - l2*focal*((rvec[1] + tvec[1])**5)*((rvec[2] + tvec[2])**(-5))
x0 = np.array([0.97, 0.22])
x = (np.array([-0.96364556,0.26718388,0.]), np.array([99825.529,-115.,45000.]), np.array([[1.5e+03,0.0e+00,5.0e+02],[0.0e+00,1.5e+03,5.0e+02],[0.0e+00,0.0e+00,1.0e+00]]), np.array([4600.32,5900.24]))
# rvec, tvec, intrinsic, xy_observed
# print(r(x))
# print(J(x))
# print(J(x).shape)

#x, Information = levenberg_marquardt(r_old, J_old, x0, Scaling=True)
#x, Information = levenberg_marquardt(r, J, x, Scaling=True)

import pickle
with open('projections.pkl', 'rb') as f:
    # Load the pickled object from the file
    projections = pickle.load(f)
with open('cameras.pkl', 'rb') as f:
    # Load the pickled object from the file
    cameras = pickle.load(f)
with open('points.pkl', 'rb') as f:
    # Load the pickled object from the file
    points = pickle.load(f)
    
input_data_raw = [] 
for proj_index in range(0,len(projections)):
    single_proj_data_0 = (cameras[projections[proj_index][0]][0],cameras[projections[proj_index][0]][1],cameras[projections[proj_index][0]][2],projections[proj_index][1:3])
    input_data_raw.append(single_proj_data_0)
    single_proj_data_1 = (cameras[projections[proj_index][3]][0],cameras[projections[proj_index][3]][1],cameras[projections[proj_index][3]][2],projections[proj_index][4:6])
    input_data_raw.append(single_proj_data_1)
    
def r(x):
    residuals = []
    
    for i in range(0,len(x)):
        r = np.zeros((2,))
        rvec, tvec, camera, xy_obs = x[i]
        p0 = rvec[0] + tvec[0]
        p1 = rvec[1] + tvec[1]
        p2 = rvec[2] + tvec[2]
        xp = -p0/p2
        yp = -p1/p2
        r2 = xp**2 + yp**2
        l1 = 1
        l2 = 1
        distortion = 1.0 + r2*(l1 + l2*r2)
        focal = camera[0][0]
        predicted_x = focal*distortion*xp
        predicted_y = focal*distortion*yp
        
        r[0] = predicted_x - xy_obs[0]
        r[1] = predicted_y - xy_obs[1]
        residuals.append(r[0])
        residuals.append(r[1])
    residuals = np.array(residuals)
    
    return residuals

def J(x):
    jacobians = []
    
    for i in range(0,len(x)):
        # J = np.zeros((2, 7))
        J_0 = np.zeros((7,))
        J_1 = np.zeros((7,))
        rvec, tvec, camera, _ = x[i]
        l1 = 1
        l2 = 1
        #l1,l2 берем константой
        focal = camera[0][0]
        
        # dfocal pred_x
        J_0[0] = -(rvec[0] + tvec[0])*((rvec[2] + tvec[2])**(-1)) - l1*((rvec[0] + tvec[0])**3)*((rvec[2] + tvec[2])**(-3)) - l1*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-3)) - l2*((rvec[0] + tvec[0])**5)*((rvec[2] + tvec[2])**(-5)) - 2*l2*((rvec[0] + tvec[0])**3)*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-5)) - l2*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**4)*((rvec[2] + tvec[2])**(-5))
        # drvec[0] pred_x
        J_0[1] = -focal*(1)*((rvec[2] + tvec[2])**(-1)) - 3*l1*focal*((rvec[0] + tvec[0])**2)*((rvec[2] + tvec[2])**(-3)) - l1*focal*(1)*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-3)) - 5*l2*focal*((rvec[0] + tvec[0])**4)*((rvec[2] + tvec[2])**(-5)) - 6*l2*focal*((rvec[0] + tvec[0])**2)*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-5)) - l2*focal*(1)*((rvec[1] + tvec[1])**4)*((rvec[2] + tvec[2])**(-5))
        # drvec[1] pred_x
        J_0[2] = -2*l1*focal*(rvec[0] + tvec[0])*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-3)) - 4*l2*focal*((rvec[0] + tvec[0])**3)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-5)) - 4*l2*focal*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-5))
        # drvec[2] pred_x
        J_0[3] = focal*(rvec[0] + tvec[0])*(1/(rvec[2] + tvec[2])**(2)) + 3*l1*focal*((rvec[0] + tvec[0])**3)*((rvec[2] + tvec[2])**(-4)) + 3*l1*focal*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-4)) + 5*l2*focal*((rvec[0] + tvec[0])**5)*((rvec[2] + tvec[2])**(-6)) + 10*l2*focal*((rvec[0] + tvec[0])**3)*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-6)) + 5*l2*focal*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**4)*((rvec[2] + tvec[2])**(-6))
        # dtvec[0] pred x
        J_0[4] = -focal*(1)*((rvec[2] + tvec[2])**(-1)) - 3*l1*focal*((rvec[0] + tvec[0])**2)*((rvec[2] + tvec[2])**(-3)) - l1*focal*(1)*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-3)) - 5*l2*focal*((rvec[0] + tvec[0])**4)*((rvec[2] + tvec[2])**(-5)) - 6*l2*focal*((rvec[0] + tvec[0])**2)*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-5)) - l2*focal*(1)*((rvec[1] + tvec[1])**4)*((rvec[2] + tvec[2])**(-5))
        # dtvec[1] pred x
        J_0[5] = -2*l1*focal*(rvec[0] + tvec[0])*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-3)) - 4*l2*focal*((rvec[0] + tvec[0])**3)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-5)) - 4*l2*focal*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-5))
        # dtvec[2] pred x
        J_0[6] = focal*(rvec[0] + tvec[0])*(1/(rvec[2] + tvec[2])**(2)) + 3*l1*focal*((rvec[0] + tvec[0])**3)*((rvec[2] + tvec[2])**(-4)) + 3*l1*focal*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-4)) + 5*l2*focal*((rvec[0] + tvec[0])**5)*((rvec[2] + tvec[2])**(-6)) + 10*l2*focal*((rvec[0] + tvec[0])**3)*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-6)) + 5*l2*focal*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**4)*((rvec[2] + tvec[2])**(-6))
        
        # dfocal pred_y
        J_1[0] = -(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-1)) - l1*((rvec[0] + tvec[0])**2)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-3)) - l1*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-3)) - l2*((rvec[0] + tvec[0])**4)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-5)) - 2*l2*((rvec[0] + tvec[0])**2)*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-5)) - l2*((rvec[1] + tvec[1])**5)*((rvec[2] + tvec[2])**(-5))
        # drvec[0] pred_y
        J_1[1] = -2*l1*focal*(rvec[0] + tvec[0])*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-3)) - 4*l2*focal*((rvec[0] + tvec[0])**3)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-5)) - 4*l2*focal*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-5))
        # drvec[1] pred_y
        J_1[2] = -focal*(1)*((rvec[2] + tvec[2])**(-1)) - l1*focal*((rvec[0] + tvec[0])**2)*(1)*((rvec[2] + tvec[2])**(-3)) - 3*l1*focal*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-3)) - l2*focal*((rvec[0] + tvec[0])**4)*(1)*((rvec[2] + tvec[2])**(-5)) - 6*l2*focal*((rvec[0] + tvec[0])**2)*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-5)) - 5*l2*focal*((rvec[1] + tvec[1])**4)*((rvec[2] + tvec[2])**(-5))
        # drvec[2] pred_y
        J_1[3] = focal*(rvec[1] + tvec[1])*(1/(rvec[2] + tvec[2])**(2)) + 3*l1*focal*((rvec[0] + tvec[0])**2)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-4)) + 3*l1*focal*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-4)) + 5*l2*focal*((rvec[0] + tvec[0])**4)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-6)) + 10*l2*focal*((rvec[0] + tvec[0])**2)*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-6)) + 5*l2*focal*((rvec[1] + tvec[1])**5)*((rvec[2] + tvec[2])**(-6))
        # dtvec[0] pred_y
        J_1[4] = -2*l1*focal*(rvec[0] + tvec[0])*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-3)) - 4*l2*focal*((rvec[0] + tvec[0])**3)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-5)) - 4*l2*focal*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-5))
        # dtvec[1] pred_y
        J_1[5] = -focal*(1)*((rvec[2] + tvec[2])**(-1)) - l1*focal*((rvec[0] + tvec[0])**2)*(1)*((rvec[2] + tvec[2])**(-3)) - 3*l1*focal*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-3)) - l2*focal*((rvec[0] + tvec[0])**4)*(1)*((rvec[2] + tvec[2])**(-5)) - 6*l2*focal*((rvec[0] + tvec[0])**2)*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-5)) - 5*l2*focal*((rvec[1] + tvec[1])**4)*((rvec[2] + tvec[2])**(-5))
        # dtvec[2] pred_y
        J_1[6] = focal*(rvec[1] + tvec[1])*(1/(rvec[2] + tvec[2])**(2)) + 3*l1*focal*((rvec[0] + tvec[0])**2)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-4)) + 3*l1*focal*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-4)) + 5*l2*focal*((rvec[0] + tvec[0])**4)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-6)) + 10*l2*focal*((rvec[0] + tvec[0])**2)*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-6)) + 5*l2*focal*((rvec[1] + tvec[1])**5)*((rvec[2] + tvec[2])**(-6))
        # Расчеты матрицы Якоби выполнялись вследствие следующих упрощений
        # distortion = 1.0 + l1*(xp**2 + yp**2) + l2*(xp**2 + yp**2)**2
        # predicted_x = focal*xp + focal*xp*l1*(xp**2 + yp**2) + focal*xp*l2*(xp**2 + yp**2)**2
        # predicted_y = focal*yp + focal*yp*l1*(xp**2 + yp**2) + focal*yp*l2*(xp**2 + yp**2)**2
        # predicted_x = focal*(xp) + focal*l1*(xp**3) + focal*l1*(xp)*(yp**2) + focal*l2*(xp**5) + 2*focal*l2*(xp**3)*(yp**2) + focal*l2*(xp)*(yp**4)
        # predicted_y = focal*(yp) + focal*(yp)*l1*(xp**2) + focal*l1*(yp**3) + focal*(yp)*l2*(xp**4) + 2*focal*l2*(xp**2)*(yp**3) + focal*l2*(yp**5)
        # predicted_x = -focal*p0*p2**(-1) - focal*l1*p0**3*p2**(-3) - focal*l1*p0*p2**(-1)*p1**2*p2**(-2) - focal*l2*p0**5*p2**(-5) - 2*focal*l2*p0**3*p2**(-3)*p1**2*p2**(-2) - focal*l2*p0*p2**(-1)*p1**4*p2**(-4)
        # predicted_y = -focal*p1*p2**(-1) - focal*p1*p2**(-1)*l1*p0**2*p2**(-2) - focal*l1*p1**3*p2**(-3) - focal*p1*p2(-1)*l2*p0**4*p2**(-4) - 2*focal*l2*p0**2*p2**(-2)*p1**3*p2**(-3) - focal*l2*p1**5*p2**(-5)
        # predicted_x = -focal*p0*(p2**(-1)) - l1*focal*(p0**3)*(p2**(-3)) - l1*focal*p0*(p1**2)*(p2**(-3)) - l2*focal*(p0**5)*(p2**(-5)) - 2*l2*focal*(p0**3)*(p1**2)*(p2**(-5)) - l2*focal*p0*(p1**4)*(p2**(-5))
        # predicted_y = -focal*p1*(p2**(-1)) - l1*focal*(p0**2)*p1*(p2**(-3)) - l1*focal*(p1**3)*(p2**(-3)) - l2*focal*(p0**4)*p1*(p2**(-5)) - 2*l2*focal*(p0**2)*(p1**3)*(p2**(-5)) - l2*focal*(p1**5)*(p2**(-5))
        # predicted_x = -focal*(rvec[0] + tvec[0])*((rvec[2] + tvec[2])**(-1)) - l1*focal*((rvec[0] + tvec[0])**3)*((rvec[2] + tvec[2])**(-3)) - l1*focal*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-3)) - l2*focal*((rvec[0] + tvec[0])**5)*((rvec[2] + tvec[2])**(-5)) - 2*l2*focal*((rvec[0] + tvec[0])**3)*((rvec[1] + tvec[1])**2)*((rvec[2] + tvec[2])**(-5)) - l2*focal*(rvec[0] + tvec[0])*((rvec[1] + tvec[1])**4)*((rvec[2] + tvec[2])**(-5))
        # predicted_y = -focal*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-1)) - l1*focal*((rvec[0] + tvec[0])**2)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-3)) - l1*focal*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-3)) - l2*focal*((rvec[0] + tvec[0])**4)*(rvec[1] + tvec[1])*((rvec[2] + tvec[2])**(-5)) - 2*l2*focal*((rvec[0] + tvec[0])**2)*((rvec[1] + tvec[1])**3)*((rvec[2] + tvec[2])**(-5)) - l2*focal*((rvec[1] + tvec[1])**5)*((rvec[2] + tvec[2])**(-5))
        jacobians.append(J_0)
        jacobians.append(J_1)
    jacobians = np.array(jacobians)
    return jacobians



x, Information = levenberg_marquardt(r, J, input_data_raw, Scaling=True)