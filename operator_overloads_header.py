from typing import Dict, List, TextIO


def constructor_header(
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
    outfile.write(f"  __host__ __device__ complex_{new_base_type}(void);\n")
    for base_type in base_types:
        outfile.write(f"  __host__ __device__ complex_{new_base_type}(const {base_type} x);\n")
    for base_type1 in base_types:
        for base_type2 in base_types:
            outfile.write(
                f"  __host__ __device__ complex_{new_base_type}(const {base_type1} a, const {base_type2} b);\n"
            )
    for new_type in new_types:
        # Write decleration for contructors that take a high priority and contructs a low priority
        if priority[new_type] > priority[f"complex_{new_base_type}"]:
            outfile.write(f"  __host__ __device__ complex_{new_base_type}(const {new_type} &x);\n")
            continue
        if priority[new_type] == priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(f"  __host__ __device__ complex_{new_base_type}(const {new_type} &x);\n")


def overload_negate_header(
    new_base_type: str,
    outfile: TextIO,
) -> None:
    """Overloads the -(void) operator.

    Args:
      new_base_type: base type for the new complex struct.
      outfile: File pointer to write struct to.
    """
    outfile.write(f"  __host__ __device__ complex_{new_base_type} operator-(void) const;\n")


def overload_equal_header(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
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
        outfile.write(f"  __host__ __device__ complex_{new_base_type}& operator=(const {base_type} x);\n")
    for new_type in new_types:
        outfile.write(f"  __host__ __device__ complex_{new_base_type}& operator=(const {new_type} &x);\n")


def overload_plusequal_header(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
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
        outfile.write(f"  __host__ __device__ complex_{new_base_type}& operator+=(const {base_type} x);\n")
    for new_type in new_types:
        outfile.write(f"  __host__ __device__ complex_{new_base_type}& operator+=(const {new_type} &x);\n")


def overload_minusequal_header(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
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
        outfile.write(f"  __host__ __device__ complex_{new_base_type}& operator-=(const {base_type} x);\n")
    for new_type in new_types:
        outfile.write(f"  __host__ __device__ complex_{new_base_type}& operator-=(const {new_type} &x);\n")


def overload_prodequal_header(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
    outfile: TextIO,
) -> None:
    """Overload the *= operator.

    Args:
      new_base_type: base type for the new complex struct.
      base_types: base types for which a complex corrospondance should be contructed.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
      outfile: File pointer to write struct to.
    """
    for base_type in base_types:
        outfile.write(f"  __host__ __device__ complex_{new_base_type}& operator*=(const {base_type} x);\n")
    for new_type in new_types:
        outfile.write(f"  __host__ __device__ complex_{new_base_type}& operator*=(const {new_type} &x);\n")


def overload_divequal_header(
    new_base_type: str,
    base_types: List[str],
    new_types: List[str],
    outfile: TextIO,
) -> None:
    """Overload the /= operator.

    Args:
      new_base_type: base type for the new complex struct.
      base_types: base types for which a complex corrospondance should be contructed.
      new_types: new complex types.
      priority: Priority for resulting typer when operating two different types.
      outfile: File pointer to write struct to.
    """
    for base_type in base_types:
        outfile.write(f"  __host__ __device__ complex_{new_base_type}& operator/=(const {base_type} x);\n")
    for new_type in new_types:
        outfile.write(f"  __host__ __device__ complex_{new_base_type}& operator/=(const {new_type} &x);\n")


def overload_plus_rhs_header(
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
            f"  __host__ __device__ complex_{new_base_type} operator+(const {base_type} x) const;\n"
        )
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] > priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(
            f"  __host__ __device__ complex_{new_base_type} operator+(const {new_type} &x) const;\n"
        )


def overload_minus_rhs_header(
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
            f"  __host__ __device__ complex_{new_base_type} operator-(const {base_type} x) const;\n"
        )
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] > priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(
            f"  __host__ __device__ complex_{new_base_type} operator-(const {new_type} &x) const;\n"
        )


def overload_product_rhs_header(
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
            f"  __host__ __device__ complex_{new_base_type} operator*(const {base_type} x) const;\n"
        )
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] > priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(
            f"  __host__ __device__ complex_{new_base_type} operator*(const {new_type} &x) const;\n"
        )


def overload_division_rhs_header(
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
            f"  __host__ __device__ complex_{new_base_type} operator/(const {base_type} x) const;\n"
        )
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] > priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(
            f"  __host__ __device__ complex_{new_base_type} operator/(const {new_type} &x) const;\n"
        )


def overload_plus_lhs_header(
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
            f"__host__ __device__ complex_{new_base_type} operator+(const {base_type} x, const complex_{new_base_type} &c);\n"
        )
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] >= priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(
            f"__host__ __device__ complex_{new_base_type} operator+(const {new_type} &x, const complex_{new_base_type} &c);\n"
        )


def overload_minus_lhs_header(
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
            f"__host__ __device__ complex_{new_base_type} operator-(const {base_type} x, const complex_{new_base_type} &c);\n"
        )
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] >= priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(
            f"__host__ __device__ complex_{new_base_type} operator-(const {new_type} &x, const complex_{new_base_type} &c);\n"
        )


def overload_product_lhs_header(
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
            f"__host__ __device__ complex_{new_base_type} operator*(const {base_type} x, const complex_{new_base_type} &c);\n"
        )
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] >= priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(
            f"__host__ __device__ complex_{new_base_type} operator*(const {new_type} &x, const complex_{new_base_type} &c);\n"
        )


def overload_division_lhs_header(
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
            f"__host__ __device__ complex_{new_base_type} operator/(const {base_type} x, const complex_{new_base_type} &c);\n"
        )
    for new_type in new_types:
        # Only write this operator for types of lover priority
        if priority[new_type] >= priority[f"complex_{new_base_type}"]:
            continue
        outfile.write(
            f"__host__ __device__ complex_{new_base_type} operator/(const {new_type} &x, const complex_{new_base_type} &c);\n"
        )
