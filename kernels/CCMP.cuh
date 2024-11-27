
__global__ void AES_CCMP_Encrypt(char *State, char *CipherKey, const unsigned int StateLength,
                                 char *grcon, char *gsbox, char *gmul2, char *gmul3,
                                 unsigned int nonce, unsigned int counter) {
    int index = (threadIdx.x + blockDim.x * blockIdx.x) * 16;

    // Cargar tablas de búsqueda en memoria compartida
    __shared__ char rcon[256], sbox[256], mul2[256], mul3[256];
    if (threadIdx.x < 256) {
        rcon[threadIdx.x] = grcon[threadIdx.x];
        sbox[threadIdx.x] = gsbox[threadIdx.x];
        mul2[threadIdx.x] = gmul2[threadIdx.x];
        mul3[threadIdx.x] = gmul3[threadIdx.x];
    }
    __syncthreads();

    // Expansión de clave
    __shared__ char ExpandedKey[16 * (NR_ROUNDS + 1)];
    if (threadIdx.x == 0) {
        KeyExpansion(CipherKey, ExpandedKey, rcon, sbox);
    }
    __syncthreads();

    // Encriptación
    if (index + 16 <= StateLength) {
        char stateLocal[16];
        char counterBlock[16];

        // Cargar bloque de State
        for (int i = 0; i < 16; i++) {
            stateLocal[i] = State[index + i];
        }

        // Preparar bloque contador
        memset(counterBlock, 0, 16);
        *(unsigned int *) &counterBlock[0] = nonce;
        *(unsigned int *) &counterBlock[12] = counter + (index / 16);

        // Cifrar bloque contador
        AddRoundKey(counterBlock, ExpandedKey);
        for (int r = 1; r < NR_ROUNDS; r++)
            Round(counterBlock, ExpandedKey + 16 * r, sbox, mul2, mul3);
        FinalRound(counterBlock, ExpandedKey + 16 * NR_ROUNDS, sbox);

        // XOR con State para cifrar
        for (int i = 0; i < 16; i++) {
            stateLocal[i] ^= counterBlock[i];
        }

        // Escribir resultado cifrado
        for (int i = 0; i < 16; i++) {
            State[index + i] = stateLocal[i];
        }
    }
}


__global__ void AES_CCMP_Decrypt(char *State, char *CipherKey, const unsigned int StateLength,
                                 char *grcon, char *gsbox, char *gmul2, char *gmul3,
                                 unsigned int nonce, unsigned int counter) {
    int index = (threadIdx.x + blockDim.x * blockIdx.x) * 16;

    // Cargar tablas de búsqueda en memoria compartida
    __shared__ char rcon[256], sbox[256], mul2[256], mul3[256];
    if (threadIdx.x < 256) {
        rcon[threadIdx.x] = grcon[threadIdx.x];
        sbox[threadIdx.x] = gsbox[threadIdx.x];
        mul2[threadIdx.x] = gmul2[threadIdx.x];
        mul3[threadIdx.x] = gmul3[threadIdx.x];
    }
    __syncthreads();

    // Expansión de clave
    __shared__ char ExpandedKey[16 * (NR_ROUNDS + 1)];
    if (threadIdx.x == 0) {
        KeyExpansion(CipherKey, ExpandedKey, rcon, sbox);
    }
    __syncthreads();

    // Desencriptación
    if (index + 16 <= StateLength) {
        char stateLocal[16];
        char counterBlock[16];

        // Cargar bloque de State (texto cifrado)
        for (int i = 0; i < 16; i++) {
            stateLocal[i] = State[index + i];
        }

        // Preparar bloque contador
        memset(counterBlock, 0, 16);
        *(unsigned int *) &counterBlock[0] = nonce;
        *(unsigned int *) &counterBlock[12] = counter + (index / 16);

        // Cifrar bloque contador (igual que en la encriptación)
        AddRoundKey(counterBlock, ExpandedKey);
        for (int r = 1; r < NR_ROUNDS; r++)
            Round(counterBlock, ExpandedKey + 16 * r, sbox, mul2, mul3);
        FinalRound(counterBlock, ExpandedKey + 16 * NR_ROUNDS, sbox);

        // XOR con State para descifrar
        for (int i = 0; i < 16; i++) {
            stateLocal[i] ^= counterBlock[i];
        }

        // Escribir resultado descifrado
        for (int i = 0; i < 16; i++) {
            State[index + i] = stateLocal[i];
        }
    }
}