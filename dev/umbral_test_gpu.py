import random
import numpy as np
from pycuda import driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
from typing import List, Tuple

# Primo de 384 bits (usado en la curva P-384):
FIELD_SIZE = 2 ** 384 - 2 ** 128 - 2 ** 96 + 2 ** 32 - 1

# Número de partes de 64 bits necesarias para representar nuestros números
NUM_PARTS = (384 + 63) // 64

# Código CUDA
cuda_code = """
#define NUM_PARTS 6

__device__ int compare(unsigned long long *a, unsigned long long *b) {
    for (int i = NUM_PARTS - 1; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

__device__ void mod_add(unsigned long long *a, unsigned long long *b, unsigned long long *result, unsigned long long *m) {
    unsigned long long carry = 0;
    for (int i = 0; i < NUM_PARTS; i++) {
        unsigned long long sum = a[i] + b[i] + carry;
        result[i] = sum;
        carry = sum < a[i] || (sum == a[i] && carry);
    }

    // Reducción modular
    if (carry || compare(result, m) >= 0) {
        unsigned long long borrow = 0;
        for (int i = 0; i < NUM_PARTS; i++) {
            unsigned long long diff = result[i] - m[i] - borrow;
            result[i] = diff;
            borrow = (diff > result[i]) || (diff == result[i] && borrow);
        }
    }
}

__device__ void mod_mul(unsigned long long *a, unsigned long long *b, unsigned long long *result, unsigned long long *m) {
    unsigned long long temp[NUM_PARTS * 2] = {0};
    for (int i = 0; i < NUM_PARTS; i++) {
        unsigned long long carry = 0;
        for (int j = 0; j < NUM_PARTS; j++) {
            unsigned long long prod_low = a[i] * b[j];
            unsigned long long prod_high = __umul64hi(a[i], b[j]);
            unsigned long long sum = prod_low + temp[i+j] + carry;
            temp[i+j] = sum;
            carry = prod_high + (sum < prod_low);
        }
        temp[i + NUM_PARTS] = carry;
    }

    // Reducción modular (simplificada para este ejemplo)
    for (int i = 0; i < NUM_PARTS; i++) {
        result[i] = temp[i];
    }
}

__global__ void evaluate_polynomial(unsigned long long *coeffs, int degree, unsigned long long *x_values, 
                                    unsigned long long *results, int n, unsigned long long *field_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long result[NUM_PARTS] = {0};
        unsigned long long x_pow[NUM_PARTS];
        for (int i = 0; i < NUM_PARTS; i++) {
            x_pow[i] = (i == 0) ? x_values[idx * NUM_PARTS + i] : 0;
        }

        for (int i = 0; i <= degree; i++) {
            unsigned long long term[NUM_PARTS];
            mod_mul(&coeffs[i * NUM_PARTS], x_pow, term, field_size);
            mod_add(result, term, result, field_size);
            if (i < degree) {
                mod_mul(x_pow, &x_values[idx * NUM_PARTS], x_pow, field_size);
            }
        }

        for (int i = 0; i < NUM_PARTS; i++) {
            results[idx * NUM_PARTS + i] = result[i];
        }
    }
}
"""

# Compilar el módulo CUDA
mod = SourceModule(cuda_code, options=['-diag-suppress', '63'])
evaluate_polynomial_gpu = mod.get_function("evaluate_polynomial")


def int_to_parts(x: int) -> List[int]:
    return [x & ((1 << 64) - 1)] + [
        (x >> (64 * i)) & ((1 << 64) - 1)
        for i in range(1, NUM_PARTS)
    ]


def parts_to_int(parts: List[int]) -> int:
    return sum(int(part) << (64 * i) for i, part in enumerate(parts))


def generate_secret() -> int:
    return random.randint(1, FIELD_SIZE - 1)


def generate_polynomial(secret: int, degree: int) -> List[int]:
    return [secret] + [random.randint(1, FIELD_SIZE - 1) for _ in range(degree)]


def generate_shares(secret: int, num_shares: int, threshold: int, start_index: int = 1) -> List[Tuple[int, int]]:
    poly = generate_polynomial(secret, threshold - 1)
    x_values = np.array([int_to_parts(x) for x in range(start_index, start_index + num_shares)], dtype=np.uint64)

    # Preparar datos para GPU
    coeffs_gpu = cuda.mem_alloc(len(poly) * NUM_PARTS * 8)
    x_values_gpu = cuda.mem_alloc(num_shares * NUM_PARTS * 8)
    results_gpu = cuda.mem_alloc(num_shares * NUM_PARTS * 8)
    field_size_gpu = cuda.mem_alloc(NUM_PARTS * 8)

    cuda.memcpy_htod(coeffs_gpu, np.array([int_to_parts(coeff) for coeff in poly], dtype=np.uint64))
    cuda.memcpy_htod(x_values_gpu, x_values)
    cuda.memcpy_htod(field_size_gpu, np.array(int_to_parts(FIELD_SIZE), dtype=np.uint64))

    # Ejecutar kernel
    evaluate_polynomial_gpu(
        coeffs_gpu, np.int32(threshold - 1), x_values_gpu, results_gpu,
        np.int32(num_shares), field_size_gpu,
        block=(256, 1, 1), grid=((num_shares + 255) // 256, 1)
    )

    # Obtener resultados
    results = np.empty((num_shares, NUM_PARTS), dtype=np.uint64)
    cuda.memcpy_dtoh(results, results_gpu)

    # Convert results to Python integers
    return list(zip(range(start_index, start_index + num_shares),
                    [parts_to_int([int(x) for x in result]) for result in results]))


def generate_shares_cpu(secret: int, num_shares: int, threshold: int, start_index: int = 1) -> List[Tuple[int, int]]:
    poly = generate_polynomial(secret, threshold - 1)
    return [(x, sum((coeff * pow(x, power, FIELD_SIZE)) % FIELD_SIZE for power, coeff in enumerate(poly)) % FIELD_SIZE)
            for x in range(start_index, start_index + num_shares)]


def reconstruct_secret(shares: List[Tuple[int, int]], threshold: int) -> int:
    if len(shares) < threshold:
        raise ValueError("No hay suficientes sombras para reconstruir el secreto.")

    def lagrange_interpolation(x: int, shares: List[Tuple[int, int]]) -> int:
        total = 0
        for i, share_i in shares[:threshold]:
            numerator, denominator = 1, 1
            for j, _ in shares[:threshold]:
                if i != j:
                    numerator = (numerator * (x - j)) % FIELD_SIZE
                    denominator = (denominator * (i - j)) % FIELD_SIZE
            lagrange_polynomial = (share_i * numerator * pow(denominator, FIELD_SIZE - 2, FIELD_SIZE)) % FIELD_SIZE
            total = (total + lagrange_polynomial) % FIELD_SIZE
        return total

    return lagrange_interpolation(0, shares)


def compare_shares(gpu_shares, cpu_shares):
    for i, (gpu_share, cpu_share) in enumerate(zip(gpu_shares, cpu_shares)):
        if gpu_share != cpu_share:
            print(f"Diferencia en el share {i}:")
            print(f"  GPU: {gpu_share}")
            print(f"  CPU: {cpu_share}")
            return False
    return True


def main():
    secret = generate_secret()
    initial_participants = 5
    threshold = 2

    print(f"Secreto original: {secret}")
    print(f"Participantes iniciales: {initial_participants}")
    print(f"Umbral: {threshold}")

    shares_gpu = generate_shares(secret, initial_participants, threshold)
    shares_cpu = generate_shares_cpu(secret, initial_participants, threshold)

    print("\nComparando todos los shares generados:")
    if compare_shares(shares_gpu, shares_cpu):
        print("Todos los shares son idénticos entre GPU y CPU.")
    else:
        print("Se encontraron diferencias en los shares.")

    print("\nSombras generadas para los participantes iniciales.")
    print("Comparación de las primeras 5 sombras (GPU vs CPU):")
    for i in range(5):
        print(f"GPU: {shares_gpu[i]}, CPU: {shares_cpu[i]}")

    new_participants = [1, 2, 5]

    for num_new in new_participants:
        current_participants = len(shares_gpu)
        new_shares_gpu = generate_shares(secret, num_new, threshold, start_index=current_participants + 1)
        new_shares_cpu = generate_shares_cpu(secret, num_new, threshold, start_index=current_participants + 1)
        shares_gpu.extend(new_shares_gpu)
        shares_cpu.extend(new_shares_cpu)

        print(f"\nAñadidos {num_new} nuevos participantes. Total actual: {len(shares_gpu)}")

        pair_gpu = random.sample(shares_gpu, threshold)
        pair_cpu = random.sample(shares_cpu, threshold)
        print(f"Par de sombras aleatorias para reconstruir el secreto (GPU): {pair_gpu}")
        print(f"Par de sombras aleatorias para reconstruir el secreto (CPU): {pair_cpu}")

        reconstructed_secret_gpu = reconstruct_secret(pair_gpu, threshold)
        reconstructed_secret_cpu = reconstruct_secret(pair_cpu, threshold)

        print(f"Secreto reconstruido (GPU): {reconstructed_secret_gpu}")
        print(f"Secreto reconstruido (CPU): {reconstructed_secret_cpu}")
        print(f"Secreto original: {secret}")
        print(f"¿Reconstrucción exitosa (GPU)? {'Sí' if reconstructed_secret_gpu == secret else 'No'}")
        print(f"¿Reconstrucción exitosa (CPU)? {'Sí' if reconstructed_secret_cpu == secret else 'No'}")

        if reconstructed_secret_gpu != secret:
            print("Analizando los shares usados para la reconstrucción en GPU:")
            for x, y in pair_gpu:
                print(f"  x: {x}, y: {y}")


if __name__ == "__main__":
    main()