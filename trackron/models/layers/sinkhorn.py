import torch


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    norm = - (ms + ns).log()
    log_mu = norm.expand(m)
    log_nu = norm.expand(n)
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(scores, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def ot_optimize(C, max_iter=100, eps=1.0):
    b,m,n = C.shape
    # both marginals are fixed with equal weights
    mu = torch.empty(b, m, dtype=torch.float,
                     requires_grad=False).fill_(1.0 / m).squeeze().cuda()
    nu = torch.empty(b, m, dtype=torch.float,
                     requires_grad=False).fill_(1.0 / n).squeeze().cuda()

    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)
    # To check if algorithm terminates because of threshold
    # or max iterations reached
    # Stopping criterion
    thresh = 1e-8

    def modi_cost(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (c_{ij} + u_i + v_j) / \epsilon$"
        return (C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    # Sinkhorn iterations
    for i in range(max_iter):
        u1 = u  # useful to check the update
        u = eps * (torch.log(mu+1e-8) -
                   torch.logsumexp(modi_cost(C, u, v), dim=-1)) + u
        v = eps * \
            (torch.log(nu+1e-8) -
                torch.logsumexp(modi_cost(C, u, v).transpose(-2, -1), dim=-1)) + v
        err = (u - u1).abs().sum(-1).mean()

        if err.item() < thresh:
            break

    U, V = u, v
    # Transport plan pi = diag(a)*K*diag(b)
    pi = torch.exp(modi_cost(C, U, V))
    return pi
