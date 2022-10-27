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
            task_theta = []
            task_grads = []
            for p in group['params']:
                # each p represents the tensor object of one layer
                if p.grad is not None:  # shape: [T,shape of parameters]
                    task_grads.append(p.grad)
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
            gdash_layered_list = []

            # after expanding step 10
            for task_index in range(task_num):
                g_list = grads_tasks_list[task_index]  # represent the gradient calculated on each layer.
                gdash_layered_list = self._scaler_product(g_list, alpha[task_index])

            # step 10
            for index_task in range(task_num):
                g_list = grads_tasks_list[index_task]
                sum1 = self._product_grads(g_list, gdash_layered_list)
                task_t1.append(sum1)

            minimum_tensor = torch.kthvalue(torch.tensor(task_t1), 1)  # step 10
            t_chosen = minimum_tensor.indices.item()  # step 10, getting the index from kthvalue method.

            # step 11
            g_list_chosen = grads_tasks_list[t_chosen]  # this g is for a particular task, chosen by t_chosen
            gamma_t = self.find_gamma(gdash_layered_list, g_list_chosen)  # step 11

            # do a scaler product between gamma and gdash
            scaled_product = self._scaler_product(gdash_layered_list, gamma_t)
            # step 11 ends here
            calculated_gamma = self._product_grads(scaled_product, scaled_product)  # not sure if this works
            # if calculated_gamma <= 0.0001:
            #     return alpha
            print(f'the current value of gamma is: {calculated_gamma}')
            alpha = torch.mul(alpha, (1 - calculated_gamma))  # step 12 first part
            alpha[t_chosen] += calculated_gamma  # step 12 second part

            # alpha = torch.sum(torch.mul(alpha, (1-calculated_gamma)), torch.mul(calculated_gamma, unit_chosen))
        print(f'the current value of gamma is: {calculated_gamma}')
        return alpha

    def _scaler_product(self, g_list: list, alpha: float) -> list:
        """
        Returns the aggregated sum of multiplication of alpha with gradient.
        Used in step 10 and step 11 of algor 2 of paper 1.
        Args:
            g_list: each index of the list represents the layer, and the value is the tensor for that layer.
            alpha: refer to algorithm 2, step 7.

        Returns:
            a list object with as many indexes as there are layers in the network.
        """

        gdash_dict = dict()
        for layer_index in range(len(g_list)):
            if gdash_dict.__contains__(layer_index):
                gdash_dict[layer_index] += torch.mul(alpha, g_list[layer_index])
            else:
                gdash_dict[layer_index] = torch.mul(alpha, g_list[layer_index])

        return list(gdash_dict.values())

    def _product_grads(self, g_list: list, gdash_list: list) -> float:
        """
        This method multiplies both parameters such that a scaler value is returned.
        Both parameters have same shape.

        Args:
            g_list: list of tensors, with each index representing a gradient of layer of neural network.
            gdash_list: list of tensors with each tensor representing a gradient of layer of neural network.

        Returns:
            sum as in step 10 of algorithm 2.
        """

        sum1 = 0
        for index_layer in range(len(gdash_list)):
            g = g_list[index_layer]
            gdash = gdash_list[index_layer]
            sum1 += torch.matmul(torch.flatten(g).T, torch.flatten(gdash))

        return sum1

    def _subtract_list_tensors(self, tensor_list1: list, tensor_list2: list) -> list:
        """
        Subtracts the two different list of tensors element by element.

        Args:
            tensor_list1: contains as many elements as there are layers in the neural network. Each tensor is a gradient.
            tensor_list2: contains as many elements as there are layers in the neural network. Each tensor is a gradient.

        Returns:
            list of subtracted tensors.
        """

        output_list = []
        for layer in range(len(tensor_list1)):
            output_list.append(torch.sub(tensor_list1[layer], tensor_list2[layer]))

        return output_list

    def find_gamma(self, g_dash_list: list, g_list: list):
        """
        Use Algorithm 1 from the paper. Both parameters are of same shape and are list of gradients. Each layer
        corresponds to a layer in the neural network.

        Args:
            g_dash_list: list of tensors, each element represent a gradient for the layer.
            g_list: list of tensors, each element is gradient for the layer.

        Returns:
            scaler value
        """

        product1 = self._product_grads(g_list, g_dash_list)
        product2 = self._product_grads(g_list, g_list)
        product3 = self._product_grads(g_dash_list, g_dash_list)
        product4 = self._product_grads(self._subtract_list_tensors(g_dash_list, g_list), g_dash_list)
        product5 = self._product_grads(self._subtract_list_tensors(g_list, g_dash_list),
                                       self._subtract_list_tensors(g_list, g_dash_list))

        if product1 >= product2:
            return 1
        elif product1 >= product3:
            return 0
        else:
            return product4/product5
