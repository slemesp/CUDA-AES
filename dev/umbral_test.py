import random
from decimal import getcontext
from typing import List, Tuple

# Aumentamos la precisión de Decimal
getcontext().prec = 1000

# Primo de 256 bits (el que estabas usando):
FIELD_SIZE = 2 ** 256 - 2 ** 224 + 2 ** 192 + 2 ** 96 - 1

# Primo de Mersenne de 521 bits (usado en criptografía de curva elíptica):
FIELD_SIZE = 2 ** 521 - 1


# Primo de 384 bits (usado en la curva P-384):
FIELD_SIZE = 2**384 - 2**128 - 2**96 + 2**32 - 1

# Primo de 224 bits (usado en la curva P-224):
# FIELD_SIZE = 2**224 - 2**96 + 1

# Primo de 192 bits (usado en la curva P-192):
# FIELD_SIZE = 2**192 - 2**64 - 1

# Primo de 160 bits (usado en algunas implementaciones de DSA):
# FIELD_SIZE = 2**160 - 2**31 - 1

# Primo de 128 bits:
# FIELD_SIZE = 2**128 - 159


# Elegimos uno de los primos (puedes cambiar esto según necesites)
# FIELD_SIZE = 2 ** 256 - 2 ** 224 + 2 ** 192 + 2 ** 96 - 1

def generate_secret() -> int:
    """Genera un secreto aleatorio."""
    return random.randint(1, FIELD_SIZE - 1)


def generate_polynomial(secret: int, degree: int) -> List[int]:
    """Genera un polinomio aleatorio de grado 'degree' con el secreto como término constante."""
    return [secret] + [random.randint(1, FIELD_SIZE - 1) for _ in range(degree)]


def evaluate_polynomial(poly: List[int], x: int) -> int:
    """Evalúa el polinomio en x."""
    return sum((coef * pow(x, power, FIELD_SIZE)) % FIELD_SIZE for power, coef in enumerate(poly)) % FIELD_SIZE


def generate_shares(secret: int, num_shares: int, threshold: int, start_index: int = 1) -> List[Tuple[int, int]]:
    """Genera 'num_shares' sombras para el secreto con un umbral de 'threshold'."""
    poly = generate_polynomial(secret, threshold - 1)
    return [(i, evaluate_polynomial(poly, i)) for i in range(start_index, start_index + num_shares)]


def mod_inverse(a: int, m: int) -> int:
    """Calcula el inverso modular de a módulo m usando el algoritmo extendido de Euclides."""

    def egcd(a: int, b: int) -> Tuple[int, int, int]:
        if a == 0:
            return (b, 0, 1)
        else:
            g, y, x = egcd(b % a, a)
            return (g, x - (b // a) * y, y)

    g, x, _ = egcd(a, m)
    if g != 1:
        raise Exception('El inverso modular no existe')
    return x % m


def reconstruct_secret(shares: List[Tuple[int, int]], threshold: int) -> int:
    """Reconstruye el secreto a partir de 'threshold' sombras."""
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
            lagrange_polynomial = (share_i * numerator * mod_inverse(denominator, FIELD_SIZE)) % FIELD_SIZE
            total = (total + lagrange_polynomial) % FIELD_SIZE
        return total

    return lagrange_interpolation(0, shares)


def main():
    # Inicialización
    secret = generate_secret()
    initial_participants = 50
    threshold = 2

    print(f"Secreto original: {secret}")
    print(f"Participantes iniciales: {initial_participants}")
    print(f"Umbral: {threshold}")

    # Generar sombras iniciales
    shares = generate_shares(secret, initial_participants, threshold)

    print("\nSombras generadas para los participantes iniciales.")

    # Simular la adición de nuevos participantes
    new_participants = [1, 2, 5]  # Añadiremos 1, luego 2, luego 5 participantes

    for num_new in new_participants:
        current_participants = len(shares)
        new_shares = generate_shares(secret, num_new, threshold, start_index=current_participants + 1)
        shares.extend(new_shares)

        print(f"\nAñadidos {num_new} nuevos participantes. Total actual: {len(shares)}")

        pair = random.sample(shares, threshold)
        print(f"Par de sombras aleatorias para reconstruir el secreto: {pair}")
        reconstructed_secret = reconstruct_secret(pair, threshold)

        print(f"Secreto reconstruido con {threshold} sombras aleatorias: {reconstructed_secret}")
        print(f"¿Reconstrucción exitosa? {'Sí' if reconstructed_secret == secret else 'No'}")


if __name__ == "__main__":
    main()
