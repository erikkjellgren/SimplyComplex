from write_complex_struct import write_headerfile

if __name__ == "__main__":
    types = ["int", "float", "double"]
    base_priority = {}
    for i, typ in enumerate(types):
        base_priority[typ] = i + 1
    write_headerfile(types, base_priority)

    with open("gpu_complex.h.tmp", "r", encoding="UTF-8") as file:
        header_template = file.readlines()

    with open("SimplyComplex.h", "r", encoding="UTF-8") as file:
        complex_struct = file.readlines()

    with open("gpu_complex.h", "w", encoding="UTF-8") as new_header:
        write = True
        for line in header_template:
            if "#define I (hr_complex_int(0, 1))" in line:
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

    with open("SimplyComplex.cu", "r", encoding="UTF-8") as file:
        complex_struct_members = file.readlines()

    with open("gpu_complex.c", "w", encoding="UTF-8") as new_cudafile:
        new_cudafile.write("/***************************************************************************\\\n")
        new_cudafile.write("* Copyright (c) 2008, Claudio Pica                                          *\n")
        new_cudafile.write("* All rights reserved.                                                      *\n")
        new_cudafile.write("\\***************************************************************************/\n")
        new_cudafile.write("\n")
        new_cudafile.write(
            "/*******************************************************************************\n"
        )
        new_cudafile.write("*\n")
        new_cudafile.write("* File gpu_complex.c\n")
        new_cudafile.write("*\n")
        new_cudafile.write("* Type definitions and macros for complex numbers used in C++ and CUDA\n")
        new_cudafile.write("*\n")
        new_cudafile.write(
            "*******************************************************************************/\n"
        )
        new_cudafile.write("\n")
        new_cudafile.write(
            "/*******************************************************************************\n"
        )
        new_cudafile.write("*\n")
        new_cudafile.write("* Definitions of type complex\n")
        new_cudafile.write("* The following structs are generated using:\n")
        new_cudafile.write("* https://github.com/erikkjellgren/SimplyComplex\n")
        new_cudafile.write("* Do NOT change them manually\n")
        new_cudafile.write("*\n")
        new_cudafile.write(
            "*******************************************************************************/\n"
        )
        new_cudafile.write('#include "hr_complex.h"\n')
        new_cudafile.write("#ifdef WITH_GPU\n")
        for struct_line in complex_struct_members:
            if "SimplyComplex.h" in struct_line:
                continue
            new_cudafile.write(
                struct_line.replace("complex_int", "hr_complex_int")
                .replace("complex_float", "hr_complex_flt")
                .replace("complex_double", "hr_complex")
            )
        new_cudafile.write("#endif //WITH_GPU\n")
