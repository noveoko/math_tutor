import random, numpy as np
from scipy.optimize import minimize as _scipy_minimize

def _sigma(z):
    return np.where(z >= 0, 1./(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def _bce_loss_and_grad(params, y, student_idx, kc_idx, opp, n_students, n_kcs):
    beta  = params[:n_students]
    delta = params[n_students:n_students+n_kcs]
    gamma = params[n_students+n_kcs:]
    z = beta[student_idx] + delta[kc_idx] + gamma[kc_idx] * opp
    p = _sigma(z)
    loss_vec = np.maximum(z,0) - y*z + np.log1p(np.exp(-np.abs(z)))
    loss = float(np.mean(loss_vec))
    residual = (p - y) / len(y)
    grad_beta = np.zeros(n_students); grad_delta = np.zeros(n_kcs); grad_gamma = np.zeros(n_kcs)
    np.add.at(grad_beta, student_idx, residual)
    np.add.at(grad_delta, kc_idx, residual)
    np.add.at(grad_gamma, kc_idx, residual * opp)
    return loss, np.concatenate([grad_beta, grad_delta, grad_gamma])

rng = random.Random(42)
evs = [{'u': f's{s}', 'opp': opp, 'y': 1 if rng.random() < min(0.95, 0.3+0.07*opp) else 0}
       for s in range(30) for opp in range(10)]
sv = {s:i for i,s in enumerate(sorted(set(e['u'] for e in evs)))}
n_s, n_k = len(sv), 1
s_arr = np.array([sv[e['u']] for e in evs], dtype=np.int32)
kc_arr = np.zeros(len(evs), dtype=np.int32)
opp_arr = np.array([e['opp'] for e in evs], dtype=np.float64)
y_arr = np.array([e['y'] for e in evs], dtype=np.float64)
x0 = np.zeros(n_s + 2*n_k)
res = _scipy_minimize(_bce_loss_and_grad, x0, method='L-BFGS-B', jac=True,
    args=(y_arr, s_arr, kc_arr, opp_arr, n_s, n_k),
    options={'maxiter': 1000, 'ftol': 1e-9})
import scipy
print('scipy:', scipy.__version__)
print('success:', res.success, '| status:', res.status, '| message:', res.message)
