if __name__ == "__main__":
    with open("gpu_complex.h.tmp", "r", encoding="UTF-8") as file:
        header_template = file.readlines()

    with open("SimplyComplex.h", "r", encoding="UTF-8") as file:
        complex_struct = file.readlines()

    with open("gpu_complex.h", "w", encoding="UTF-8") as new_header:
        write = True
        for line in header_template:
            if "#define I (hr_complex(0.0, 1.0))" in line:
                write = True
                new_header.write(
                    "/*******************************************************************************\n"
                )
                new_header.write("*\n")
                new_header.write("* End of generated code\n")
                new_header.write("*\n")
                new_header.write(
                    "*******************************************************************************/\n"
                )
            if write:
                new_header.write(line)
            if "* Definitions of type complex" in line:
                write = False
                new_header.write("* The following structs are generated using:\n")
                new_header.write("* https://github.com/erikkjellgren/SimplyComplex\n")
                new_header.write("* Do NOT change them manually\n")
                new_header.write("*\n")
                new_header.write(
                    "*******************************************************************************/\n"
                )
                for struct_line in complex_struct:
                    new_header.write(
                        struct_line.replace("complex_int", "hr_complex_int")
                        .replace("complex_float", "hr_complex_flt")
                        .replace("complex_double", "hr_complex")
                    )
