import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model

def model_A(x, params):
    y = params[0] + x*params[1] + params[2]*x**2
    return y
    
def model_B(x, params):
    y = params[0]*(np.exp(-0.5*(x-params[1])**2/params[2]**2))
    return y

def model_C(x,params):
    y = params[0]*(np.exp(-0.5*(x-params[1])**2/params[2]**2))
    y += params[0]*(np.exp(-0.5*(x-params[3])**2/params[4]**2))
    return y

def loglike(x_obs, y_obs, sigma_y_obs, betas, model):
       
    n_obs = len(y_obs)
    l = 0.0
    for i in range(n_obs):
        l += -0.5*(y_obs[i]-model(x_obs[i,:], betas))**2/sigma_y_obs[i]**2
    return l
    
def run_mcmc(data_file, n_dim, n_iterations, model, escala):
    
    data = np.loadtxt(data_file)
    x_obs = data[:,:1]
    y_obs = data[:, 1]
    sigma_y_obs = data[:,-1]

    betas = np.zeros([n_iterations, n_dim+1])
    for i in range(1, n_iterations):
        current_betas = betas[i-1,:]
        next_betas = current_betas + np.random.normal(scale=escala, size=n_dim+1)

        loglike_current = loglike(x_obs, y_obs, sigma_y_obs, current_betas, model)
        loglike_next = loglike(x_obs, y_obs, sigma_y_obs, next_betas, model)

        r = np.min([np.exp(loglike_next - loglike_current), 1.0])
        alpha = np.random.random()

        if alpha < r:
            betas[i,:] = next_betas
        else:
            betas[i,:] = current_betas
    betas = betas[-10000:,:]
    return {'betas':betas, 'x_obs':x_obs, 'y_obs':y_obs, 'sigma_y':sigma_y_obs}

def graficar(n_dim, betas, nombre, model, results, BIC):
    
    plt.figure(figsize=(18,3))
    
    for i in range(0,n_dim+1):
        plt.subplot(1,n_dim+2,i+1)
        plt.hist(betas[:,i],bins=15, density=True)
        plt.title(r"$\beta_{}= {:.4f}\pm {:.4f}$".format(i,np.mean(betas[:,i]), np.std(betas[:,i])))
        plt.xlabel(r"$\beta_{}$".format(i))
        
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(1,n_dim+2,n_dim+2)
    plt.errorbar(results['x_obs'], results['y_obs'], yerr=results['sigma_y'], fmt='none', alpha = 0.5)
    plt.scatter(results['x_obs'], model(results['x_obs'], np.mean(betas, axis = 0)), c ='r', label = 'mode', s=2)
    plt.scatter(results['x_obs'], results['y_obs'], s = 2)
    plt.title('BIC = %s'%str(BIC))
    plt.savefig(nombre,  bbox_inches='tight')  
    
results_A = run_mcmc("data_to_fit.txt", 2,200000, model_A, 0.01)
betas_A = results_A['betas']
B_C_MA_2 = -loglike(results_A['x_obs'], results_A['y_obs'], results_A['sigma_y'], np.mean(betas_A, axis = 0), model_A) + 3/2*np.log(31)
graficar(2, betas_A, 'resultado_A.png', model_A, results_A,B_C_MA_2)


results_B = run_mcmc("data_to_fit.txt", 2,30000, model_B,0.1)
betas_B = results_B['betas']
B_C_MB_2 = -loglike(results_B['x_obs'], results_B['y_obs'], results_B['sigma_y'], np.mean(betas_B, axis = 0), model_B) + 3/2*np.log(31)
graficar(2, betas_B, 'resultado_B.png', model_B, results_B,B_C_MB_2)

results_C = run_mcmc("data_to_fit.txt", 4,50000, model_C,0.1)
betas_C = results_C['betas']
B_C_MC_2 = -loglike(results_C['x_obs'], results_C['y_obs'], results_C['sigma_y'], np.mean(betas_C, axis = 0), model_C) + 5/2*np.log(31)
graficar(4, betas_C, 'resultado_C.png', model_C, results_C,B_C_MC_2)



