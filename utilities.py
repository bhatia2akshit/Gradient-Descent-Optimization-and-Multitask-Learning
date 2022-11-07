import torch


def aggregated_scaler_product(list1: list, list2: torch.tensor) -> list:
    """
    Returns the aggregated sum of multiplication of an element of list1 with the element of list2.
    Used in step 10 and step 11 of expansion algor 2 of paper 1.

    :math:`\\sum_{t=1}^T\\alpha^t*g\_list^t`


    Args:
        list1: each element of the list represents the list of n elements. In algo 2, g_list is list1.
        list2: list with the length same as list1. In algo 2, alpha is list2.

    Returns:
        a list object with as many indexes as there are layers in the network.
    """

    gdash_dict = dict()
    for task_index in range(len(list1)):
        product_list = scaler_product(list1[task_index], list2[task_index])
        for layer in range(len(product_list)):
            if gdash_dict.__contains__(layer):
                gdash_dict[layer] += product_list[layer]
            else:
                gdash_dict[layer] = product_list[layer]

    return list(gdash_dict.values())


def scaler_product(grad_list: list, alpha: float) -> list:
    """
    Returns the multiplication of alpha with gradients.
    Used in step 10 and step 11 of algor 2 of paper 1.

    :math:`[\\alpha * grad]`


    Args:
        grad_list: each index of the list represents the layer, and the value is the tensor for that layer.
        alpha: refer to algorithm 2, step 7.

    Returns:
        a list object with as many indexes as there are layers in the network.
    """

    product_list = list()
    for layer_index in range(len(grad_list)):
        product_list.append(torch.mul(alpha, grad_list[layer_index]))

    return product_list


def product_grads(g_list: list, gdash_list: list) -> float:
    """
    This method multiplies both parameters such that a scaler value is returned.
    Both parameters have same shape.

    :math:`<\\nabla f_r(\\theta), \\sum_{t=1}^T\\alpha^t*g\_list^t >`

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


def subtract_list_tensors(tensor_list1: list, tensor_list2: list) -> list:
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


def find_gamma(g_dash_list: list, g_list: list):
    """
    Use Algorithm 1 from the paper. Both parameters are of same shape and are list of gradients. Each layer
    corresponds to a layer in the neural network.

    Args:
        g_dash_list: list of tensors, each element represent a gradient for the layer.
        g_list: list of tensors, each element is gradient for the layer.

    Returns:
        scaler value
    """

    product1 = product_grads(g_list, g_dash_list)
    product2 = product_grads(g_list, g_list)
    product3 = product_grads(g_dash_list, g_dash_list)
    product4 = product_grads(subtract_list_tensors(g_dash_list, g_list), g_dash_list)
    product5 = product_grads(subtract_list_tensors(g_list, g_dash_list), subtract_list_tensors(g_list, g_dash_list))

    if product1 >= product2:
        return 1
    elif product1 >= product3:
        return 0
    else:
        return product4/product5

