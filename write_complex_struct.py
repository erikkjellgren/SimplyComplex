from typing import Dict, List, TextIO

import numpy as np

from operator_overloads import (
    constructor,
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


def run_writer(base_types: List[str], base_priority_: Dict[str, int]) -> None:
    """Construct needed tracking to the complex struct.

    Args:
      base_types: base types for which a complex corrospondance should be contructed.
      base_priority_: the priority of the base types for resulting type when used together.
    """
    tmp = []
    for priority_value in base_priority_.values():
        if not isinstance(priority_value, int):
            raise TypeError(f"base_priority values must be integers; type: {type(priority_value)}")
        if priority_value < 1:
            raise ValueError(f"base_priority values must be above zero; value: {priority_value}")
        if priority_value in tmp:
            raise ValueError(f"All base_priority values must be different; duplicate value: {priority_value}")
        tmp.append(priority_value)

    new_types = []
    priority = base_priority_.copy()
    max_priority = np.max(list(base_priority_.values()))
    for base_type in base_types:
        type_name = f"complex_{base_type}"
        new_types.append(type_name)
        # Only guarenteed to be a non-exsisting priority_value because
        # priority values are required to be above zero.
        priority[type_name] = max_priority + priority[base_type]
    write_file(base_types, new_types, priority)


def write_struct(
    base_types: List[str],
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    """Write the struct and calls the writing of the operator overloads.

    Args:
      base_types: base types for which a complex corrospondance should be contructed.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
      outfile: File pointer to write struct to.
    """
    for base_type in base_types:
        outfile.write(f"struct complex_{base_type}")
        outfile.write("{\n")
        outfile.write(f"  {base_type} re, im\n")
        constructor(base_type, base_types, new_types, priority, outfile)
        overload_equal(base_type, base_types, new_types, priority, outfile)
        overload_negate(base_type, outfile)
        overload_plusequal(base_type, base_types, new_types, priority, outfile)
        overload_minusequal(base_type, base_types, new_types, priority, outfile)
        overload_prodequal(base_type, base_types, new_types, priority, outfile)
        overload_divequal(base_type, base_types, new_types, priority, outfile)
        overload_plus_rhs(base_type, base_types, new_types, priority, outfile)
        overload_minus_rhs(base_type, base_types, new_types, priority, outfile)
        overload_product_rhs(base_type, base_types, new_types, priority, outfile)
        overload_division_rhs(base_type, base_types, new_types, priority, outfile)
        outfile.write("};\n")
    for base_type in base_types:
        overload_plus_lhs(base_type, base_types, new_types, priority, outfile)
        overload_minus_lhs(base_type, base_types, new_types, priority, outfile)
        overload_product_lhs(base_type, base_types, new_types, priority, outfile)
        overload_division_lhs(base_type, base_types, new_types, priority, outfile)
    for base_type in base_types:
        overload_equal_reverse_priority(base_type, new_types, priority, outfile)
        overload_plusequal_reverse_priority(base_type, new_types, priority, outfile)
        overload_minusequal_reverse_priority(base_type, new_types, priority, outfile)
        overload_prodequal_reverse_priority(base_type, new_types, priority, outfile)
        overload_divequal_reverse_priority(base_type, new_types, priority, outfile)


def write_file(base_types: List[str], new_types: List[str], priority: Dict[str, int]) -> None:
    """Create the headerfile, and controls the writing that is around structs.

    Args:
      base_types: base types for which a complex corrospondance should be contructed.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
    """
    with open("SimplyComplex.h", "w", encoding="UTF-8") as headerfile:
        # Write dummy structs of higher priority
        for new_type in new_types:
            headerfile.write(f"struct {new_type};\n")
        write_struct(base_types, new_types, priority, headerfile)


if __name__ == "__main__":
    types = ["int", "float", "double"]
    base_priority = {}
    for i, typ in enumerate(types):
        base_priority[typ] = i + 1
    run_writer(types, base_priority)
