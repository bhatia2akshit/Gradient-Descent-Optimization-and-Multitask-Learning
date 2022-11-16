# writing optimizer method suitable for pytorch
import torch
import utilities

class FrankWolfOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, batch_size=1, max_iter=10):
        """
            params: list of parameters that are updated (usually a list of tensor objects is given)
        """

        if lr < 0:
            raise ValueError('lr must be greater than or equal to 0.')
        defaults = dict(lr=lr, batch_size=batch_size, max_iter=max_iter)
        # self.task_list = []
        self.task_theta = []
        self.task_grads = []
        super(FrankWolfOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):  # look at the changing of name
        for group in self.param_groups:
            task_theta = []
            task_grads = []
            for p in group['params']:
                # each p represents the tensor object of one layer
                if p.grad is not None:  # shape: [T,shape of parameters]
                    task_grads.append(p.grad.clone())
                    task_theta.append(p)

            self.task_theta.append(task_theta)
            self.task_grads.append(task_grads)

    def frank_wolf_solver(self):
        """
        Compute FrankWolf Solver as referenced in Algorithm 2.

        Returns:
            Tensor of t-dimensional representing alpha, with t being the number of tasks.
        """

        lr = self.param_groups[0]['lr']
        task_num = self.param_groups[0]['batch_size']
        max_iter = self.param_groups[0]['max_iter']
        grads_tasks_list = self.task_grads  # each element contains all grads for every layer for each mini batch

        alpha = torch.ones(task_num)  # shape [1,T], one alpha for each task for each batch
        alpha = torch.div(alpha, task_num)  # step 7 of algorithm 2.
        calculated_gamma = 1000
        count_iter = 0

        while count_iter < max_iter:
            task_t1 = []
            count_iter += 1

            # after expanding step 10
            gdash_layered_list = utilities.aggregated_scaler_product(grads_tasks_list, alpha)

            # step 10
            for index_task in range(task_num):
                g_list = grads_tasks_list[index_task]
                sum_product = utilities.product_grads(g_list, gdash_layered_list)
                task_t1.append(sum_product)

            minimum_tensor = torch.kthvalue(torch.tensor(task_t1), 1)  # step 10
            t_chosen = minimum_tensor.indices.item()  # step 10, getting the index from kthvalue method.

            # step 11
            g_list_chosen = grads_tasks_list[t_chosen]  # this g is for a particular task, chosen by t_chosen
            calculated_gamma = utilities.find_gamma(gdash_layered_list, g_list_chosen)  # step 11

            # do a scaler product between gamma and gdash
            # scaled_product = utilities.scaler_product(gdash_layered_list, gamma_t)
            # step 11 ends here
            # calculated_gamma = utilities.product_grads(scaled_product, scaled_product)  # not sure if this works
            if calculated_gamma <= 0.0001:
                return alpha
            print(f'the current value of calculated gamma is: {calculated_gamma}')
            alpha = torch.mul(alpha, (1 - calculated_gamma))  # step 12 first part
            alpha[t_chosen] += calculated_gamma  # step 12 second part

        print(f'the current value of gamma is: {calculated_gamma}')
        return alpha
