import typing

# this decorator must identify two methods in the class:
# 'create_world'
# infer: [RVIdentifier] -> Tensor
# The effect of this decorator is that the infer will be modified to instead call the lowered function.
# The create_world method will not be changed from the user's perspective but this decorator will update that
# create world method
def import_inference(clazz: typing.ClassVar) -> typing.ClassVar:
    """
    create new clazz that has an intercepted infer method
    """
    return clazz