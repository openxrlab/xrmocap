import json
from typing import Union


def payload_to_dict(input_instance: Union[str, dict]) -> dict:
    """Convert flask payload to python dict.

    Args:
        input_instance (Union[str, dict]):
            Payload get from request.get_json().

    Returns:
        dict: Payload in type dict.
    """
    if isinstance(input_instance, dict):
        input_dict = input_instance
    else:
        input_dict = json.loads(s=input_instance)
    return input_dict
