import os
import re

def find_function_definitions(directory):
    function_definitions = {}

    # Expresión regular para encontrar definiciones de funciones
    function_pattern = re.compile(r'(__device__|__global__)?\s*([a-zA-Z_][a-zA-Z0-9_]*\s+\**[a-zA-Z_][a-zA-Z0-9_]*\s*\(.*?\)\s*{?)')

    # Recorrer todos los archivos en el directorio
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.cuh'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        match = function_pattern.search(line)
                        if match:
                            func_signature = match.group(0).strip()
                            if file_path not in function_definitions:
                                function_definitions[file_path] = []
                            function_definitions[file_path].append(func_signature)

    return function_definitions

def main():
    directory = "../kernels"  # Cambia esto a tu ruta de archivos .cuh
    functions = find_function_definitions(directory)

    for file_path, funcs in functions.items():
        print(f"Functions in {file_path}:")
        for func in funcs:
            print(f" - {func}")

if __name__ == "__main__":
    main()