from typing import Dict, List

import numpy as np

from operator_overloads import (
    constructor,
    constructor_reverse_priority,
    overload_divequal,
    overload_divequal_reverse_priority,
    overload_division_lhs,
    overload_division_rhs,
    overload_equal,
    overload_equal_reverse_priority,
    overload_minus_lhs,
    overload_minus_rhs,
    overload_minusequal,
    overload_minusequal_reverse_priority,
    overload_negate,
    overload_plus_lhs,
    overload_plus_rhs,
    overload_plusequal,
    overload_plusequal_reverse_priority,
    overload_prodequal,
    overload_prodequal_reverse_priority,
    overload_product_lhs,
    overload_product_rhs,
)


def write_headerfile(base_types: List[str], priority: Dict[str, int]) -> None:
    """Create the headerfile, and controls the writing of the complex structs.

    Args:
      base_types: base types for which a complex corrospondance should be contructed.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
    """
    tmp = []
    for priority_value in priority.values():
        if not isinstance(priority_value, int):
            raise TypeError(f"base_priority values must be integers; type: {type(priority_value)}")
        if priority_value < 1:
            raise ValueError(f"base_priority values must be above zero; value: {priority_value}")
        if priority_value in tmp:
            raise ValueError(f"All base_priority values must be different; duplicate value: {priority_value}")
        tmp.append(priority_value)

    new_types = []
    max_priority = np.max(list(priority.values()))
    for base_type in base_types:
        type_name = f"complex_{base_type}"
        new_types.append(type_name)
        # Only guarenteed to be a non-exsisting priority_value because
        # priority values are required to be above zero.
        priority[type_name] = max_priority + priority[base_type]
    with open("SimplyComplex.h", "w", encoding="UTF-8") as headerfile:
        # Write dummy structs of higher priority
        for new_type in new_types:
            headerfile.write(f"struct {new_type};\n")
        for base_type in base_types:
            headerfile.write(f"struct __align__(16) complex_{base_type}")
            headerfile.write("{\n")
            headerfile.write(f"  {base_type} re, im;\n")
            constructor(base_type, base_types, new_types, priority, headerfile)
            overload_equal(base_type, base_types, new_types, priority, headerfile)
            overload_negate(base_type, headerfile)
            overload_plusequal(base_type, base_types, new_types, priority, headerfile)
            overload_minusequal(base_type, base_types, new_types, priority, headerfile)
            overload_prodequal(base_type, base_types, new_types, priority, headerfile)
            overload_divequal(base_type, base_types, new_types, priority, headerfile)
            overload_plus_rhs(base_type, base_types, new_types, priority, headerfile)
            overload_minus_rhs(base_type, base_types, new_types, priority, headerfile)
            overload_product_rhs(base_type, base_types, new_types, priority, headerfile)
            overload_division_rhs(base_type, base_types, new_types, priority, headerfile)
            headerfile.write("};\n")
        for base_type in base_types:
            overload_plus_lhs(base_type, base_types, new_types, priority, headerfile)
            overload_minus_lhs(base_type, base_types, new_types, priority, headerfile)
            overload_product_lhs(base_type, base_types, new_types, priority, headerfile)
            overload_division_lhs(base_type, base_types, new_types, priority, headerfile)
        for base_type in base_types:
            constructor_reverse_priority(base_type, new_types, priority, headerfile)
            overload_equal_reverse_priority(base_type, new_types, priority, headerfile)
            overload_plusequal_reverse_priority(base_type, new_types, priority, headerfile)
            overload_minusequal_reverse_priority(base_type, new_types, priority, headerfile)
            overload_prodequal_reverse_priority(base_type, new_types, priority, headerfile)
            overload_divequal_reverse_priority(base_type, new_types, priority, headerfile)


if __name__ == "__main__":
    types = ["int", "float", "double"]
    base_priority = {}
    for i, typ in enumerate(types):
        base_priority[typ] = i + 1
    write_headerfile(types, base_priority)
