# writing optimizer method suitable for pytorch
import torch
import utilities


class FrankWolfOptimizer(torch.optim.Optimizer):
    def __init__(self, params, device, lr=0.01, batch_count=1, batch_size=8, max_iter=10, ):
        """
            params: list of parameters that are updated (usually a list of tensor objects is given)
            device: type of device, GPU or CPU
            lr: float: learning rate
            batch_count: int: number of batches (tasks)
            batch_size: int: number of instances in a batch (task)
            max_iter: int: total number of iterations after which FrankWolfe method is stopped.
        """

        print(lr)
        if lr < 0:
            raise ValueError('lr must be greater than or equal to 0.')
        defaults = dict(lr=lr, max_iter=max_iter)

        self.task_theta = []
        self.grads_all_tasks = []
        self.alpha = []
        self.batch_count = batch_count
        self.batch_size = batch_size
        self.batch_counter = 0  # keeps track of which batch is it.
        self.device = device

        super(FrankWolfOptimizer, self).__init__(params, defaults)

    def collect_grads(self):  # look at the changing of name
        for group in self.param_groups:
            # task_theta = []
            task_grads = []
            for p in group['params']:
                # each p represents the tensor object of one layer
                if p.grad is not None:  # shape: [T,shape of parameters]
                    task_grads.append(p.grad.clone())

            # self.task_theta.append(task_theta)
            self.grads_all_tasks.append(task_grads)

    def step(self, closure=None):
        """
        Updates the Parameters using FrankWolfe method.
        """

        lr = self.param_groups[0]['lr']
        # task_num = self.batch_count
        alpha = self.frankwolfsolver()
        gdash_list = utilities.sum_scaled_product(self.grads_all_tasks, alpha)

        update_values = []
        for layered_index in range(len(gdash_list)):
            update_values.append(torch.mul(gdash_list[layered_index], lr))

        for index in range(len(self.param_groups[0]['params'])):
            tmp = self.param_groups[0]['params'][index].data
            self.param_groups[0]['params'][index].data = torch.sub(tmp, update_values[index])

    def frankwolfsolver(self) -> torch.tensor:
        """
        Compute FrankWolf Solver as referenced in Algorithm 2.

        Returns:
            Tensor of t-dimensional representing alpha, with t being the number of tasks.

        """
        task_num = self.batch_count
        max_iter = self.param_groups[0]['max_iter']
        grads_tasks_list = self.grads_all_tasks  # each element contains all grads for every layer for each mini batch

        alpha = torch.ones(task_num)  # shape [1,T], one alpha for each task for each batch
        alpha = torch.div(alpha, task_num)  # step 7 of algorithm 2.
        # calculated_gamma = 1000
        alpha = alpha.to(device=self.device)
        count_iter = 0

        while count_iter < max_iter:
            task_t1 = []
            count_iter += 1

            # after expanding step 10
            gdash_layered_list = utilities.sum_scaled_product(grads_tasks_list, alpha)

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

            # step 11 ends here
            if calculated_gamma <= 0.0001:
                return alpha

            alpha = torch.mul(alpha, (1 - calculated_gamma))  # step 12 first part
            alpha[t_chosen] += calculated_gamma  # step 12 second part

        return alpha
