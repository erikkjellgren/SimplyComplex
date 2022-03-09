from typing import Dict, List, TextIO


def constructor(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    """Write the constructs of the complex struct.

    Args:
      new_base_type: Base type for the new complex struct.
      base_types: Base types for which a complex corrospondance should be contructed.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
      outfile: File pointer to write struct to.
    """
    outfile.write(f"  __host__ __device__ complex_{new_base_type}(void){{}}\n")
    for base_type in base_types:
        outfile.write(f"  __host__ __device__ complex_{new_base_type}(const {base_type} x)\n")
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
    for new_type in new_types:
        if priority[new_type] == priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(f"  __host__ __device__ complex_{new_base_type}(const {new_type} x)\n")
        outfile.write("  {\n")
        outfile.write(f"    re = ({new_base_type})x.re;\n")
        outfile.write(f"    im = ({new_base_type})x.im;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")


def overload_negate(
    new_base_type: str,
    outfile: TextIO,
) -> None:
    """Overloads the -(void) operator.

    Args:
      new_base_type: base type for the new complex struct.
      outfile: File pointer to write struct to.
    """
    outfile.write(f"  __host__ __device__ complex_{new_base_type} operator-(void)\n")
    outfile.write("  {\n")
    outfile.write("    re = -re;\n")
    outfile.write("    im = -im;\n")
    outfile.write("    return *this;\n")
    outfile.write("  }\n")


def overload_equal(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    """Overloads the = operator.

    Args:
      new_base_type: base type for the new complex struct.
      base_types: base types for which a complex corrospondance should be contructed.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
      outfile: File pointer to write struct to.
    """
    for base_type in base_types:
        outfile.write(f"  __host__ __device__ complex_{new_base_type} operator=(const {base_type} x)\n")
        outfile.write("  {\n")
        outfile.write(f"    re = ({new_base_type})x;\n")
        outfile.write(f"    im = ({new_base_type})0;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] > priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(f"  __host__ __device__ complex_{new_base_type} operator=(const {new_type} x)\n")
        outfile.write("  {\n")
        outfile.write(f"    re = ({new_base_type})x.re;\n")
        outfile.write(f"    im = ({new_base_type})x.im;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")


def overload_plusequal(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    """Overload the += operator.

    Args:
      new_base_type: base type for the new complex struct.
      base_types: base types for which a complex corrospondance should be contructed.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
      outfile: File pointer to write struct to.
    """
    for base_type in base_types:
        outfile.write(f"  __host__ __device__ complex_{new_base_type} operator+=(const {base_type} x)\n")
        outfile.write("  {\n")
        outfile.write(f"    re += ({new_base_type})x;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] > priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(f"  __host__ __device__ complex_{new_base_type} operator+=(const {new_type} x)\n")
        outfile.write("  {\n")
        outfile.write(f"    re += ({new_base_type})x.re;\n")
        outfile.write(f"    im += ({new_base_type})x.im;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")


def overload_minusequal(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    """Overload the -= operator.

    Args:
      new_base_type: base type for the new complex struct.
      base_types: base types for which a complex corrospondance should be contructed.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
      outfile: File pointer to write struct to.
    """
    for base_type in base_types:
        outfile.write(f"  __host__ __device__ complex_{new_base_type} operator-=(const {base_type} x)\n")
        outfile.write("  {\n")
        outfile.write(f"    re -= ({new_base_type})x;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] > priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(f"  __host__ __device__ complex_{new_base_type} operator-=(const {new_type} x)\n")
        outfile.write("  {\n")
        outfile.write(f"    re -= ({new_base_type})x.re;\n")
        outfile.write(f"    im -= ({new_base_type})x.im;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")


def overload_prodequal(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    r"""Overload the *= operator.

    For two new complex types the multiplication is (http://www2.clarku.edu/faculty/djoyce/complex/mult.html):

    .. math::
       (x+y*i)*(u+v*i) = (x*u - y*v) + (x*v+y*u)*i

    Args:
      new_base_type: base type for the new complex struct.
      base_types: base types for which a complex corrospondance should be contructed.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
      outfile: File pointer to write struct to.
    """
    for base_type in base_types:
        outfile.write(f"  __host__ __device__ complex_{new_base_type} operator*=(const {base_type} x)\n")
        outfile.write("  {\n")
        outfile.write(f"    re *= ({new_base_type})x;\n")
        outfile.write(f"    im *= ({new_base_type})x;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] > priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(f"  __host__ __device__ complex_{new_base_type} operator*=(const {new_type} x)\n")
        outfile.write("  {\n")
        outfile.write(f"    {new_base_type} re_tmp, im_tmp;\n")
        outfile.write(f"    re_tmp = re*(({new_base_type})x.re) - im*(({new_base_type})x.im);\n")
        outfile.write(f"    im_tmp = re*(({new_base_type})x.im) + im*(({new_base_type})x.re);\n")
        outfile.write("    re = re_tmp;\n")
        outfile.write("    im = im_tmp;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")


def overload_divequal(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    r"""Overload the /= operator.

    For two new complex types the division is (http://www2.clarku.edu/faculty/djoyce/complex/div.html):

    .. math::
       \frac{x+y*i}{u+v*i} = \frac{x*u+y*v}{u*u+v*v} + \frac{y*u-x*v}{u*u+v*v}*i

    Args:
      new_base_type: base type for the new complex struct.
      base_types: base types for which a complex corrospondance should be contructed.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
      outfile: File pointer to write struct to.
    """
    for base_type in base_types:
        outfile.write(f"  __host__ __device__ complex_{new_base_type} operator/=(const {base_type} x)\n")
        outfile.write("  {\n")
        outfile.write(f"    re /= ({new_base_type})x;\n")
        outfile.write(f"    im /= ({new_base_type})x;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] > priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(f"  __host__ __device__ complex_{new_base_type} operator/=(const {new_type} x)\n")
        outfile.write("  {\n")
        outfile.write(f"    {new_base_type} re_tmp, im_tmp;\n")
        outfile.write(
            f"    re_tmp = (re*(({new_base_type})x.re) + im*(({new_base_type})x.im)) / ((({new_base_type})x.re)*(({new_base_type})x.re) + (({new_base_type})x.im)*(({new_base_type})x.im));\n"
        )
        outfile.write(
            f"    im_tmp = (im*(({new_base_type})x.re) - re*(({new_base_type})x.im)) / ((({new_base_type})x.re)*(({new_base_type})x.re) + (({new_base_type})x.im)*(({new_base_type})x.im));\n"
        )
        outfile.write("    re = re_tmp;\n")
        outfile.write("    im = im_tmp;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")


def overload_plus_rhs(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    """Overload the + operator.

    Args:
      new_base_type: base type for the new complex struct.
      base_types: base types for which a complex corrospondance should be contructed.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
      outfile: File pointer to write struct to.
    """
    for base_type in base_types:
        outfile.write(f"  __host__ __device__ complex_{new_base_type} operator+(const {base_type} x)\n")
        outfile.write("  {\n")
        outfile.write(f"    re += ({new_base_type})x;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] > priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(f"  __host__ __device__ complex_{new_base_type} operator+(const {new_type} x)\n")
        outfile.write("  {\n")
        outfile.write(f"    re += ({new_base_type})x.re;\n")
        outfile.write(f"    im += ({new_base_type})x.im;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")


def overload_minus_rhs(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    """Overload the - operator.

    Args:
      new_base_type: base type for the new complex struct.
      base_types: base types for which a complex corrospondance should be contructed.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
      outfile: File pointer to write struct to.
    """
    for base_type in base_types:
        outfile.write(f"  __host__ __device__ complex_{new_base_type} operator-(const {base_type} x)\n")
        outfile.write("  {\n")
        outfile.write(f"    re -= ({new_base_type})x;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] > priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(f"  __host__ __device__ complex_{new_base_type} operator-(const {new_type} x)\n")
        outfile.write("  {\n")
        outfile.write(f"    re -= ({new_base_type})x.re;\n")
        outfile.write(f"    im -= ({new_base_type})x.im;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")


def overload_product_rhs(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    r"""Overload the * operator.

    For two new complex types the multiplication is (http://www2.clarku.edu/faculty/djoyce/complex/mult.html):

    .. math::
       (x+y*i)*(u+v*i) = (x*u - y*v) + (x*v+y*u)*i

    Args:
      new_base_type: base type for the new complex struct.
      base_types: base types for which a complex corrospondance should be contructed.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
      outfile: File pointer to write struct to.
    """
    for base_type in base_types:
        outfile.write(f"  __host__ __device__ complex_{new_base_type} operator*(const {base_type} x)\n")
        outfile.write("  {\n")
        outfile.write(f"    re *= ({new_base_type})x;\n")
        outfile.write(f"    im *= ({new_base_type})x;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] > priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(f"  __host__ __device__ complex_{new_base_type} operator*(const {new_type} x)\n")
        outfile.write("  {\n")
        outfile.write(f"    {new_base_type} re_tmp, im_tmp;\n")
        outfile.write(f"    re_tmp = re*(({new_base_type})x.re) - im*(({new_base_type})x.im);\n")
        outfile.write(f"    im_tmp = re*(({new_base_type})x.im) + im*(({new_base_type})x.re);\n")
        outfile.write("    re = re_tmp;\n")
        outfile.write("    im = im_tmp;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")


def overload_division_rhs(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    r"""Overload the / operator.

    For two new complex types the division is (http://www2.clarku.edu/faculty/djoyce/complex/div.html):

    .. math::
       \frac{x+y*i}{u+v*i} = \frac{x*u+y*v}{u*u+v*v} + \frac{y*u-x*v}{u*u+v*v}*i

    Args:
      new_base_type: base type for the new complex struct.
      base_types: base types for which a complex corrospondance should be contructed.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
      outfile: File pointer to write struct to.
    """
    for base_type in base_types:
        outfile.write(f"  __host__ __device__ complex_{new_base_type} operator/(const {base_type} x)\n")
        outfile.write("  {\n")
        outfile.write(f"    re /= ({new_base_type})x;\n")
        outfile.write(f"    im /= ({new_base_type})x;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] > priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(f"  __host__ __device__ complex_{new_base_type} operator/(const {new_type} x)\n")
        outfile.write("  {\n")
        outfile.write(f"    {new_base_type} re_tmp, im_tmp;\n")
        outfile.write(
            f"    re_tmp = (re*(({new_base_type})x.re) + im*(({new_base_type})x.im)) / ((({new_base_type})x.re)*(({new_base_type})x.re) + (({new_base_type})x.im)*(({new_base_type})x.im));\n"
        )
        outfile.write(
            f"    im_tmp = (im*(({new_base_type})x.re) - re*(({new_base_type})x.im)) / ((({new_base_type})x.re)*(({new_base_type})x.re) + (({new_base_type})x.im)*(({new_base_type})x.im));\n"
        )
        outfile.write("    re = re_tmp;\n")
        outfile.write("    im = im_tmp;\n")
        outfile.write("    return *this;\n")
        outfile.write("  }\n")


def overload_plus_lhs(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    """Overload the + operator.

    Args:
      new_base_type: base type for the new complex struct.
      base_types: base types for which a complex corrospondance should be contructed.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
      outfile: File pointer to write struct to.
    """
    for base_type in base_types:
        outfile.write(
            f"__host__ __device__ complex_{new_base_type} operator+(const {base_type} x, complex_{new_base_type} c){{return c+x;}}\n"
        )
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] >= priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(
            f"__host__ __device__ complex_{new_base_type} operator+(const {new_type} x, complex_{new_base_type} c){{return c+x;}}\n"
        )


def overload_minus_lhs(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    """Overload the - operator.

    Args:
      new_base_type: base type for the new complex struct.
      base_types: base types for which a complex corrospondance should be contructed.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
      outfile: File pointer to write struct to.
    """
    for base_type in base_types:
        outfile.write(
            f"__host__ __device__ complex_{new_base_type} operator-(const {base_type} x, complex_{new_base_type} c){{return -c+x;}}\n"
        )
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] >= priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(
            f"__host__ __device__ complex_{new_base_type} operator-(const {new_type} x, complex_{new_base_type} c){{return -c+x;}}\n"
        )


def overload_product_lhs(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    """Overload the * operator.

    Args:
      new_base_type: base type for the new complex struct.
      base_types: base types for which a complex corrospondance should be contructed.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
      outfile: File pointer to write struct to.
    """
    for base_type in base_types:
        outfile.write(
            f"__host__ __device__ complex_{new_base_type} operator*(const {base_type} x, complex_{new_base_type} c){{return c*x;}}\n"
        )
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] >= priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(
            f"__host__ __device__ complex_{new_base_type} operator*(const {new_type} x, complex_{new_base_type} c){{return c*x;}}\n"
        )


def overload_division_lhs(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    """Overload the / operator.

    Args:
      new_base_type: base type for the new complex struct.
      base_types: base types for which a complex corrospondance should be contructed.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
      outfile: File pointer to write struct to.
    """
    for base_type in base_types:
        outfile.write(
            f"__host__ __device__ complex_{new_base_type} operator/(const {base_type} x, complex_{new_base_type} c){{return ((complex_{new_base_type})x)/c;}}\n"
        )
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] >= priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(
            f"__host__ __device__ complex_{new_base_type} operator/(const {new_type} x, complex_{new_base_type} c){{return ((complex_{new_base_type})x)/c;}}\n"
        )


def overload_equal_reverse_priority(
    new_base_type: str,
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    """Overloads the = operator, for high priority new type to lower priority new type.

    Args:
      new_base_type: base type for the new complex struct.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
      outfile: File pointer to write struct to.
    """
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] <= priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(
            f"__host__ __device__ complex_{new_base_type} complex_{new_base_type}::operator=(const {new_type} x)\n"
        )
        outfile.write("{\n")
        outfile.write(f"  re = ({new_base_type})x.re;\n")
        outfile.write(f"  im = ({new_base_type})x.im;\n")
        outfile.write("  return *this;\n")
        outfile.write("}\n")


def overload_plusequal_reverse_priority(
    new_base_type: str,
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    """Overloads the += operator, for high priority new type to lower priority new type.

    Args:
      new_base_type: base type for the new complex struct.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
      outfile: File pointer to write struct to.
    """
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] <= priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(
            f"__host__ __device__ complex_{new_base_type} complex_{new_base_type}::operator+=(const {new_type} x)\n"
        )
        outfile.write("{\n")
        outfile.write(f"  re += ({new_base_type})x.re;\n")
        outfile.write(f"  im += ({new_base_type})x.im;\n")
        outfile.write("  return *this;\n")
        outfile.write("}\n")


def overload_minusequal_reverse_priority(
    new_base_type: str,
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    """Overloads the -= operator, for high priority new type to lower priority new type.

    Args:
      new_base_type: base type for the new complex struct.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
      outfile: File pointer to write struct to.
    """
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] <= priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(
            f"__host__ __device__ complex_{new_base_type} complex_{new_base_type}::operator-=(const {new_type} x)\n"
        )
        outfile.write("{\n")
        outfile.write(f"  re -= ({new_base_type})x.re;\n")
        outfile.write(f"  im -= ({new_base_type})x.im;\n")
        outfile.write("  return *this;\n")
        outfile.write("}\n")


def overload_prodequal_reverse_priority(
    new_base_type: str,
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    r"""Overloads the *= operator, for high priority new type to lower priority new type.

    For two new complex types the division is (http://www2.clarku.edu/faculty/djoyce/complex/div.html):

    .. math::
       \frac{x+y*i}{u+v*i} = \frac{x*u+y*v}{u*u+v*v} + \frac{y*u-x*v}{u*u+v*v}*i

    Args:
      new_base_type: base type for the new complex struct.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
      outfile: File pointer to write struct to.
    """
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] <= priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(
            f"__host__ __device__ complex_{new_base_type} complex_{new_base_type}::operator*=(const {new_type} x)\n"
        )
        outfile.write("{\n")
        outfile.write(f"    {new_base_type} re_tmp, im_tmp;\n")
        outfile.write(f"    re_tmp = re*(({new_base_type})x.re) - im*(({new_base_type})x.im);\n")
        outfile.write(f"    im_tmp = re*(({new_base_type})x.im) + im*(({new_base_type})x.re);\n")
        outfile.write("    re = re_tmp;\n")
        outfile.write("    im = im_tmp;\n")
        outfile.write("    return *this;\n")
        outfile.write("}\n")


def overload_divequal_reverse_priority(
    new_base_type: str,
    new_types: List[str],
    priority: Dict[str, int],
    outfile: TextIO,
) -> None:
    r"""Overloads the /= operator, for high priority new type to lower priority new type.

    For two new complex types the division is (http://www2.clarku.edu/faculty/djoyce/complex/div.html):

    .. math::
       \frac{x+y*i}{u+v*i} = \frac{x*u+y*v}{u*u+v*v} + \frac{y*u-x*v}{u*u+v*v}*i

    Args:
      new_base_type: base type for the new complex struct.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
      outfile: File pointer to write struct to.
    """
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] <= priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(
            f"__host__ __device__ complex_{new_base_type} complex_{new_base_type}::operator/=(const {new_type} x)\n"
        )
        outfile.write("{\n")
        outfile.write(f"    {new_base_type} re_tmp, im_tmp;\n")
        outfile.write(
            f"    re_tmp = (re*(({new_base_type})x.re) + im*(({new_base_type})x.im)) / ((({new_base_type})x.re)*(({new_base_type})x.re) + (({new_base_type})x.im)*(({new_base_type})x.im));\n"
        )
        outfile.write(
            f"    im_tmp = (im*(({new_base_type})x.re) - re*(({new_base_type})x.im)) / ((({new_base_type})x.re)*(({new_base_type})x.re) + (({new_base_type})x.im)*(({new_base_type})x.im));\n"
        )
        outfile.write("    re = re_tmp;\n")
        outfile.write("    im = im_tmp;\n")
        outfile.write("    return *this;\n")
        outfile.write("}\n")
