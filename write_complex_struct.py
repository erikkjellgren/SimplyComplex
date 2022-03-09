from typing import Dict, List, TextIO

import numpy as np

from operator_overloads import (constructor, overload_divequal, overload_equal,
                                overload_minusequal, overload_plusequal,
                                overload_prodequal)


def run_writer(base_types: List[str], base_priority: Dict[str, int]) -> None:
    tmp = []
    for priority_value in base_priority.values():
        if not isinstance(priority_value, int):
            raise TypeError(
                f"base_priority values must be integers; type: {type(priority_value)}"
            )
        if priority_value < 1:
            raise ValueError(
                f"base_priority values must be above zero; value: {priority_value}"
            )
        if priority_value in tmp:
            raise ValueError(
                f"All base_priority values must be different; duplicate value: {priority_value}"
            )
        tmp.append(priority_value)

    new_types = []
    priority = base_priority.copy()
    max_priority = np.max(list(base_priority.values()))
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
    for base_type in base_types:
        outfile.write(f"struct complex_{base_type}")
        outfile.write("{\n")
        outfile.write(f"  {base_type} re, im\n")
        constructor(base_type, base_types, new_types, priority, outfile)
        overload_equal(base_type, base_types, new_types, priority, outfile)
        overload_plusequal(base_type, base_types, new_types, priority, outfile)
        overload_minusequal(base_type, base_types, new_types, priority, outfile)
        overload_prodequal(base_type, base_types, new_types, priority, outfile)
        overload_divequal(base_type, base_types, new_types, priority, outfile)
        outfile.write("};\n")
    return None


def write_file(
    base_types: List[str], new_types: List[str], priority: Dict[str, int]
) -> None:
    with open("SimplyComplex.h", "w") as headerfile:
        # Write dummy structs of higher priority
        for new_type in new_types:
            headerfile.write(f"struct {new_type};\n")
        write_struct(base_types, new_types, priority, headerfile)
    return None


if __name__ == "__main__":
    types = ["int", "float", "double"]
    priority = {}
    for i, typ in enumerate(types):
        priority[typ] = i + 1
    run_writer(types, priority)
