from typing import Dict, List, TextIO


def constructor(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    outfile.write(f"  __host__ __device__ complex_{new_base_type}(void){{}}\n")
    for base_type in base_types:
        outfile.write(
            f"  __host__ __device__ complex_{new_base_type}(const {base_type} x)\n"
        )
        outfile.write("  {\n")
        outfile.write(f"    re = ({new_base_type})x;\n")
        outfile.write(f"    im = ({new_base_type})0;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")
    for base_type1 in base_types:
        for base_type2 in base_types:
            outfile.write(
                f"  __host__ __device__ complex_{new_base_type}(const {base_type1} a, const {base_type2} b)\n"
            )
            outfile.write("  {\n")
            outfile.write(f"    re = ({new_base_type})a;\n")
            outfile.write(f"    im = ({new_base_type})b;\n")
            outfile.write("    return *this;\n")
            outfile.write("  }\n")
            if base_type1 == base_type2:
                continue
            outfile.write(
                f"  __host__ __device__ complex_{new_base_type}(const {base_type2} a, const {base_type1} b)\n"
            )
            outfile.write("  {\n")
            outfile.write(f"    re = ({new_base_type})a;\n")
            outfile.write(f"    im = ({new_base_type})b;\n")
            outfile.write("    return *this;\n")
            outfile.write("  }\n")
    """
    for new_type in new_types:
        outfile.write(f'  __host__ __device__ complex_{new_base_type}(const {new_type} x)\n')
        outfile.write('  {\n')
        outfile.write(f'    re = ({new_base_type})x.re;\n')
        outfile.write(f'    im = ({new_base_type})x.im;\n')
        outfile.write('    return *this;\n')
        outfile.write('  }\n')
    """
    return None


def overload_equal(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    for base_type in base_types:
        outfile.write(
            f"  __host__ __device__ complex_{new_base_type} operator=(const {base_type} x)\n"
        )
        outfile.write("  {\n")
        outfile.write(f"    re = ({new_base_type})x;\n")
        outfile.write(f"    im = ({new_base_type})0;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] > priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(
            f"  __host__ __device__ complex_{new_base_type} operator=(const {new_type} x)\n"
        )
        outfile.write("  {\n")
        outfile.write(f"    re = ({new_base_type})x.re;\n")
        outfile.write(f"    im = ({new_base_type})x.im;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")
    return None


def overload_plusequal(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    for base_type in base_types:
        outfile.write(
            f"  __host__ __device__ complex_{new_base_type} operator+=(const {base_type} x)\n"
        )
        outfile.write("  {\n")
        outfile.write(f"    re += ({new_base_type})x;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] > priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(
            f"  __host__ __device__ complex_{new_base_type} operator+=(const {new_type} x)\n"
        )
        outfile.write("  {\n")
        outfile.write(f"    re += ({new_base_type})x.re;\n")
        outfile.write(f"    im += ({new_base_type})x.im;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")
    return None


def overload_minusequal(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    for base_type in base_types:
        outfile.write(
            f"  __host__ __device__ complex_{new_base_type} operator-=(const {base_type} x)\n"
        )
        outfile.write("  {\n")
        outfile.write(f"    re -= ({new_base_type})x;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] > priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(
            f"  __host__ __device__ complex_{new_base_type} operator-=(const {new_type} x)\n"
        )
        outfile.write("  {\n")
        outfile.write(f"    re -= ({new_base_type})x.re;\n")
        outfile.write(f"    im -= ({new_base_type})x.im;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")
    return None


def overload_prodequal(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    for base_type in base_types:
        outfile.write(
            f"  __host__ __device__ complex_{new_base_type} operator*=(const {base_type} x)\n"
        )
        outfile.write("  {\n")
        outfile.write(f"    re *= ({new_base_type})x;\n")
        outfile.write(f"    im *= ({new_base_type})x;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] > priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(
            f"  __host__ __device__ complex_{new_base_type} operator*=(const {new_type} x)\n"
        )
        outfile.write("  {\n")
        outfile.write(f"    {new_base_type} re_tmp, im_tmp;\n")
        outfile.write("    re_tmp = re*x.re - im*x.im;\n")
        outfile.write("    im_tmp = re*x.im + im*x.re;\n")
        outfile.write("    re = re_tmp;\n")
        outfile.write("    im = im_tmp;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")
    return None


def overload_divequal(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    for base_type in base_types:
        outfile.write(
            f"  __host__ __device__ complex_{new_base_type} operator/=(const {base_type} x)\n"
        )
        outfile.write("  {\n")
        outfile.write(f"    re /= ({new_base_type})x;\n")
        outfile.write(f"    im /= ({new_base_type})x;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] > priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(
            f"  __host__ __device__ complex_{new_base_type} operator/=(const {new_type} x)\n"
        )
        outfile.write("  {\n")
        outfile.write(f"    {new_base_type} re_tmp, im_tmp;\n")
        outfile.write("    re_tmp = (re*x.re + im*x.im) / (x.re*x.re + x.im*x.im);\n")
        outfile.write("    im_tmp = (im*x.re - re*x.im) / (x.re*x.re + x.im*x.im);\n")
        outfile.write("    re = re_tmp;\n")
        outfile.write("    im = im_tmp;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")
    return None
