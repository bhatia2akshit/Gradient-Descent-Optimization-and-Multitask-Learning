import torch


def sum_scaled_product(grads_all_tasks: list, weight_tensor: torch.tensor) -> list:
    """
    Returns the aggregated sum of multiplication of an element of list1 with the element of list2.
    Used in step 10 and step 11 of expansion algor 2 of paper 1.

    :math:`\\sum_{t=1}^T\\alpha^t*g\_list^t`

    Args:
        grads_all_tasks: each element of the list represents a layer in the neural network. (g_list in algo 2 expanded.)
        weight_tensor: list value where each element is multiplied element wise to grads of each task.

    Returns:
        a list object with as many indexes as there are layers in the network.
    """

    gdash_dict = dict()
    layer_num = grads_all_tasks[0]

    for layer_index in range(len(layer_num)):
        for task_index in range(len(grads_all_tasks)):
            grads_list = grads_all_tasks[task_index]
            gdash_element = torch.mul(grads_list[layer_index], weight_tensor[task_index])
            if gdash_dict.__contains__(layer_index):
                torch.add(gdash_element, gdash_dict[layer_index], out=gdash_dict[layer_index])
            else:
                gdash_dict[layer_index] = gdash_element

    return list(gdash_dict.values())


def scaler_product(grads_layered_list: list, weight_float: float) -> list:
    """
    Returns the multiplication of alpha with gradients.
    Used in step 10 and step 11 of algor 2 of paper 1.

    :math:`[\\alpha * grad]`


    Args:
        grads_layered_list: each element of the list represents a layer in the neural network.
        weight_float: float value multiplied element wise to grads_layered_list.

    Returns:
        a list object with as many indexes as there are layers in the network.
    """

    gdash_dict = dict()

    for layer_index in range(len(grads_layered_list)):
        gdash_element_layered = torch.mul(grads_layered_list[layer_index], weight_float)
        if gdash_dict.__contains__(layer_index):
            torch.add(gdash_element_layered, gdash_dict[layer_index], out=gdash_dict[layer_index])
        else:
            gdash_dict[layer_index] = gdash_element_layered

    return list(gdash_dict.values())


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
        # sum1 += torch.matmul(torch.flatten(g).T, torch.flatten(gdash)).item()
        sum1 += torch.flatten(g) @ torch.flatten(gdash)
        # sum1 += product.item()

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

