# writing optimizer method suitable for pytorch
import torch


class FrankWolfOptimizer(torch.optim.Optimizer):
    def __init__(self, params: list, lr=1e-3, batch_size=1):
        """
            params: list of parameters that are updated (usually a list of tensor objects is given)
        """
        if lr < 0:
            raise ValueError('lr must be greater than or equal to 0.')
        defaults = dict(lr=lr, batch_size=batch_size)
        super(FrankWolfOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            params_with_grad = []  # represents the theta, [theta_sh, theta_1,...theta_M] where M is the number of
            # mini-batches

            for p in group['params']:
                if p.grad is not None:
                    #                     param_with_grads=p
                    grads = p.grad  # p.grad should be the list of gradients

                    shared_theta = p[0]
                    task_theta = p[1:]

                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                            if self.defaults['capturable'] else torch.tensor(0.)
                        state['task_grads'] = task_theta  # represent theta_t
                        state['task_shared'] = shared_theta  # represent theta_sh

                    # for task in len(range(task_theta)):
                    #     state['task_grads'][task] -= group['lr'] * grads[task+1].data  # step 2 of algorithm 2

                    # step 2
                    state['task_grads'] = torch.sub(state['task_grads'], grads[1:], alpha=group['lr'])

                    alpha = self.frank_wolf_solver(task_theta, shared_theta, grads, group['lr'], group['batch_size'],
                                                   group['max_iter'])

                    # step 5 is the dot product between alpha tensor and shared_gradient tensor (take all element of
                    # tensor as shared grad
                    grad_shared = torch.Tensor([grads[0].data]*group['batch_size'])

                    state['task_shared'] = torch.sub(state['task_shared'], torch.mul(group['lr'], torch.dot(alpha, grad_shared)))

    def frank_wolf_solver(self, task_theta, shared_theta, grads, lr, task_num, max_iter: int):
        """

        Args:
            task_theta: tensor of task specific parameters
            shared_theta: 0d tensor of  shared task parameter
            grads: 0th index is tensor of gradient wrt shared task parameter and other indices are tensors of gradients
            wrt each task parameter
            lr: learning rate
            task_num: number of tasks
            max_iter: total number of iterations after which solver would stop

        Returns:
            Tensor of t-dimensional representing alpha, with t being the number of tasks
        """
        alpha = torch.ones(task_num)
        alpha = torch.div(alpha, 1 / task_num)
        m = list()

        for grad_1 in grads[1:]:
            m.append(torch.mul(grad_1, grads[1:]))
        M = torch.stack(m)  # step 8

        minimum_tensor = torch.kthvalue(grads[1:], 1)
        gamma_chosen = 1000
        count_iter = 0
        while gamma_chosen > 0.1 or count_iter < max_iter:
            t_chosen = minimum_tensor.indices.item()  # step 10
            theta_dash = torch.dot(alpha, grads[1:])
            theta = grads[t_chosen + 1]  # plus 1 because, grads[0] is gradient for shared parameter
            gamma_t = self.find_gamma(theta_dash, theta)
            tmp = torch.sum(torch.mul(1 - gamma_t, theta_dash), torch.mul(theta, gamma_t))

            gamma_chosen = torch.square(tmp, tmp)  # step 11
            unit_chosen = torch.zeros(size=task_num)
            unit_chosen[t_chosen] = 1
            alpha = torch.sum(torch.mul(alpha, torch.sub(1, gamma_chosen)), torch.mul((gamma_chosen, unit_chosen)))

        return alpha

    def find_gamma(self, theta_dash, theta):
        """
        Args:
            theta_dash: tensor 0d
            theta: tensor 0d

        Returns:
            0d tensor
        """
        if torch.mul(theta, theta_dash) >= torch.mul(theta, theta):
            return 1
        elif torch.mul(theta, theta_dash) >= torch.mul(theta_dash, theta_dash):
            return 0
        else:
            return torch.div(torch.mul(torch.sub(theta_dash, theta), theta_dash),
                             torch.square(torch.sub(theta, theta_dash)))
