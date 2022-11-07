# writing optimizer method suitable for pytorch
import torch


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

    def step(self, closure=None):
        for group in self.param_groups:
            params_with_grad = []  # represents the theta, [theta_sh, theta_1,...theta_M] where M is the number of
            # mini-batches

            # task_dict = {}
            # task_num = len(self.task_list)
            # task_dict['task_num'] = task_num + 1  # increment the task id
            # task_dict['theta'] = []
            # task_dict['grads'] = []
            task_theta=[]
            task_grads=[]
            for p in group['params']:
                if p.grad is not None:

                    grads = p.grad  # shape: [T,shape of parameters]
                    # self.task_dict['theta'].append(p)
                    # task_dict['grads'].append(p.grad)
                    task_grads.append(p.grad)
                    task_theta.append(p)



            self.task_theta.append(task_grads)
            self.task_grads.append(task_theta)


    def frank_wolf_solver(self):
        """
        Args:
            grads: list with each element represent grads of parameters for a mini batch.
            group_param: dictionary from which learning rate, number of tasks and total number of iterations extracted.

        Returns:
            Tensor of t-dimensional representing alpha, with t being the number of tasks.
        """

        lr = self.param_groups[0]['lr']
        task_num = self.param_groups[0]['batch_size']
        max_iter = self.param_groups[0]['max_iter']
        grads = self.task_grads  # each element contains all grads for every layer for each mini batch

        alpha = torch.ones(task_num)  # shape [1,T]
        alpha = torch.div(alpha, task_num)  # step 7 of algorithm 2.
        m = list()

        # for index_i in range(task_num-1):
        #     m.append(torch.mul(grads[index_i], grads[(index_i+1):]))
        #
        # M = torch.stack(m)  # step 8

        # after expanding step 10
        gdash = {}

        for index in range(task_num):
            for index_grad in range(len(grads[index])):
                if gdash.__contains__(index_grad):
                    gdash[index_grad] += torch.mul(alpha[index], grads[index][index_grad])
                else:
                    gdash[index_grad] = torch.mul(alpha[index], grads[index][index_grad])

        task_t = []
        # there are task_num number of gradients in grads list. But gdash only has as many elements as there are layers.
        for index_task in range(task_num):
            for index_grad in range(len(gdash)):
                task_t.append(torch.matmul(grads[index_task][index_grad].T, gdash[index_grad]))  # shape [1,T]

        minimum_tensor = torch.kthvalue(task_t, 1)  # step 10
        gamma_chosen = 1000
        count_iter = 0
        while gamma_chosen > 0.1 or count_iter < max_iter:
            t_chosen = minimum_tensor.indices.item()  # step 10, getting the index from kthvalue method.
            theta_dash = torch.dot(alpha, torch.Tensor(grads))
            theta = grads[t_chosen]
            gamma_t = self.find_gamma(theta_dash, theta)  # step 11
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
