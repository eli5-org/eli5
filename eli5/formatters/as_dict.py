import attr

from .utils import numpy_to_python


def format_as_dict(explanation):
    """ Return a dictionary representing the explanation that can be JSON-encoded.
    It accepts parts of explanation (for example feature weights) as well.
    """
    return numpy_to_python(attr.asdict(explanation))


