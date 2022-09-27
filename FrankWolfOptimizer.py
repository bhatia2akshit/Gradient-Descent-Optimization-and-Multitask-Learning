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

                    grads = p.grad  # shape: [T,shape of parameters]

                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                            if self.defaults['capturable'] else torch.tensor(0.)
                        # state['task_grads'] = task_theta  # represent theta_t
                        state['task_shared'] = p  # represent theta_sh, shape: [layers_num, neuron in each layer]

                    # for task in len(range(task_theta)):
                    #     state['task_grads'][task] -= group['lr'] * grads[task+1].data  # step 2 of algorithm 2

                    # step 2
                    # state['task_grads'] = torch.sub(state['task_grads'], grads[1:], alpha=group['lr'])

                    alpha = self.frank_wolf_solver(grads, group)

                    # step 5 is the dot product between alpha tensor and shared_gradient tensor (take all element of
                    # tensor as shared grad
                    grad_shared = torch.Tensor([grads]*group['batch_size'])  # ??google p.grad.data vs p.grad??

                    state['task_shared'] = torch.sub(state['task_shared'], torch.mul(group['lr'],
                                                                                     torch.dot(alpha, grad_shared)))

    def frank_wolf_solver(self, grads, group_param: dict):
        """
        Args:
            grads: list with each element represent grads of parameters for a mini batch.
            group_param: dictionary from which learning rate, number of tasks and total number of iterations extracted.

        Returns:
            Tensor of t-dimensional representing alpha, with t being the number of tasks.
        """

        lr = group_param['lr']
        task_num = group_param['batch_size']
        max_iter = group_param['max_iter']
        alpha = torch.ones(task_num)  # shape [1,T]
        alpha = torch.div(alpha, 1 / task_num)  # step 7 of algorithm 2.
        m = list()

        # for index_i in range(task_num-1):
        #     m.append(torch.mul(grads[index_i], grads[(index_i+1):]))
        #
        # M = torch.stack(m)  # step 8

        # after expanding step 10
        gdash = torch.dot(alpha, grads)  # dot product alpha with theta.grad
        t = torch.dot(grads, gdash)  # shape [1,T]

        minimum_tensor = torch.kthvalue(t, 1)  # step 10
        gamma_chosen = 1000
        count_iter = 0
        while gamma_chosen > 0.1 or count_iter < max_iter:
            t_chosen = minimum_tensor.indices.item()  # step 10, getting the index from kthvalue method.
            theta_dash = torch.dot(alpha, grads)
            theta = grads[t_chosen]
            gamma_t = self.find_gamma(theta_dash, theta)
            tmp = torch.sum(torch.mul(1 - gamma_t, theta_dash), torch.mul(theta, gamma_t))

            gamma_chosen = torch.square(tmp)  # refer to expansion of step 11
            unit_chosen = torch.zeros(size=task_num)
            unit_chosen[t_chosen] = 1
            alpha = torch.sum(torch.mul(alpha, (1-gamma_chosen)), torch.mul((gamma_chosen, unit_chosen)))

        return alpha

    def find_gamma(self, theta_dash: torch.tensor, theta: torch.tensor):
        """
        Args:
            theta_dash: tensor [1,shape of neural network]
            theta: tensor [shape of neural network]

        Returns:
            0d tensor
        """
        if torch.dot(torch.transpose(theta), theta_dash) >= torch.dot(torch.transpose(theta), theta):
            return 1
        elif torch.dot(torch.transpose(theta), theta_dash) >= torch.mul(torch.transpose(theta_dash), theta_dash):
            return 0
        else:
            return torch.div(torch.mul(torch.transpose(torch.sub(theta_dash, theta)), theta_dash),
                             torch.square(torch.sub(theta, theta_dash)))
