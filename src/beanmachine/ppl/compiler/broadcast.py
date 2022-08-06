import math
import typing
from typing import Callable, List

from torch import Size

identity_fnc = lambda a: a


def broadcast_fnc(input_size: Size, target_size: Size) -> typing.Union[bool, Callable]:
    if input_size == target_size:
        return identity_fnc

    # Make the input size length equal to target size by buffering with 1's
    input_project_size = []
    ones_to_add = len(target_size) - len(input_size)
    for _ in range(0, ones_to_add):
        input_project_size.append(1)
    for dim in input_size:
        input_project_size.append(dim)

    assert len(input_project_size) == len(target_size)

    # the input can be broadcast to the target if
    # input_dim[i] == target_dim[i] || input_dim[i] == 1 for all i
    for i in range(0, len(target_size)):
        if input_project_size[i] != 1 and target_size[i] != input_project_size[i]:
            return False

    # in order to map from a composite index to a coordinate index we
    # need to know how many elements are in each element of each dimension
    # for example, in the case of a list of matrices we might have the size 4 x 3 x 2
    # which means we have a list of 4 elements, where each element is a matrix of 6 elements.
    # Within the matrix, we have 3 elements, each of size 2. In this case, the group size array
    # should be [6, 2, 1]
    group_size = []
    current = 1
    L = len(target_size)
    for k in range(0, L).__reversed__():
        d = target_size[k]
        group_size.append(current)
        current = current * d

    # given a global index, produce a coordinate
    def target_index_to_composite(ti: int) -> List:
        index_list = []
        current_index = ti
        j = len(target_size) - 1
        for _ in target_size:
            next_index = math.floor(current_index / group_size[j])
            index_list.append(next_index)
            current_index = current_index % group_size[j]
            j = j - 1
        return index_list

    # product list should be [2, 1, 1]
    product_list = []
    current = 1
    # the element at index N-j should be the size of the group at dimension j
    # for [1,1,3] we want [1,3,3]. For [3,2,1] we want [1,1,2]
    for k in range(0, len(input_project_size)).__reversed__():
        d = input_project_size[k]
        product_list.append(current)
        current = current * d

    # given a coordinate index of target, compute a global index of input
    def input_list_from_target_list(target_list: List[int]) -> int:
        i = 0
        j = len(product_list) - 1
        index = 0
        for inx in target_list:
            if input_project_size[i] == 1:
                i = i + 1
                j = j - 1
                continue
            else:
                next = inx * product_list[j]
                index = index + next
            j = j - 1
            i = i + 1
        return index

    return lambda target_index: input_list_from_target_list(
        target_index_to_composite(target_index)
    )
