#define AES_PRIVATESTATE_SHAREDLUT
#define LUT_IN_SHARED
#define NR_ROUNDS 10


__device__ void SubBytes(char *block, char *sbox) {
    for (unsigned int i = 0; i < 16; i++) {
        block[i] = sbox[(unsigned char) block[i]];
    }
}


__device__ void ShiftRows(char *block) {
    // ROW 0: Untouched

    // ROW 1: Shift left by one
    char temp = block[1];
    block[1] = block[5];
    block[5] = block[9];
    block[9] = block[13];
    block[13] = temp;

    // ROW 2: Shift left by two
    temp = block[2];
    block[2] = block[10];
    block[10] = temp;
    temp = block[6];
    block[6] = block[14];
    block[14] = temp;

    // ROW 3: Shift left by three
    temp = block[15];
    block[15] = block[11];
    block[11] = block[7];
    block[7] = block[3];
    block[3] = temp;
}


__device__ char xtime(unsigned char poly) {
    char out = poly << 1;
    if (!(out & 0x80)) {
        // polynomial not yet in reduced form
        out ^= 0x1b;
    }
    return out;
}


__device__ void mixColumns(char *block, char *mul2, char *mul3) {
    char temp[16];
    for (unsigned int col = 0; col < 4; col++) {
        temp[4 * col] = mul2[(unsigned char) block[4 * col]] ^ mul3[(unsigned char) block[4 * col + 1]] ^ block[
                            4 * col + 2] ^ block[4 * col + 3];
        temp[4 * col + 1] = block[4 * col] ^ mul2[(unsigned char) block[4 * col + 1]] ^ mul3[(unsigned char) block[
                                4 * col + 2]] ^ block[4 * col + 3];
        temp[4 * col + 2] = block[4 * col] ^ block[4 * col + 1] ^ mul2[(unsigned char) block[4 * col + 2]] ^ mul3[(
                                unsigned char) block[4 * col + 3]];
        temp[4 * col + 3] = mul3[(unsigned char) block[4 * col]] ^ block[4 * col + 1] ^ block[4 * col + 2] ^ mul2[(
                                unsigned char) block[4 * col + 3]];
    }

    for (unsigned int i = 0; i < 16; i++) {
        block[i] = temp[i];
    }
}

__device__ void AddRoundKey(char *block, char *RoundKey) {
    block[0] ^= RoundKey[0];
    block[1] ^= RoundKey[1];
    block[2] ^= RoundKey[2];
    block[3] ^= RoundKey[3];
    block[4] ^= RoundKey[4];
    block[5] ^= RoundKey[5];
    block[6] ^= RoundKey[6];
    block[7] ^= RoundKey[7];
    block[8] ^= RoundKey[8];
    block[9] ^= RoundKey[9];
    block[10] ^= RoundKey[10];
    block[11] ^= RoundKey[11];
    block[12] ^= RoundKey[12];
    block[13] ^= RoundKey[13];
    block[14] ^= RoundKey[14];
    block[15] ^= RoundKey[15];
}


__device__ void Round(char *block, char *roundkey, char *sbox, char *mul2, char *mul3) {
    SubBytes(block, sbox);
    ShiftRows(block);
    mixColumns(block, mul2, mul3);
    AddRoundKey(block, roundkey);
}


__device__ void RotByte(char *array) {
    char temp = array[0];
    array[0] = array[1];
    array[1] = array[2];
    array[2] = array[3];
    array[3] = temp;
}


__device__ void SubByte(char *array, char *sbox) {
    // Need to convert to unsigned char for correct indexing
    array[0] = sbox[(unsigned char) array[0]];
    array[1] = sbox[(unsigned char) array[1]];
    array[2] = sbox[(unsigned char) array[2]];
    array[3] = sbox[(unsigned char) array[3]];
}

__device__ void KeyExpansion(char *CipherKey, char *ExpandedKey, char *rcon, char *sbox) {
    // First part of the expanded key is equal to the Cipher key.
    for (int i = 0; i < 16; i++)
        ExpandedKey[i] = CipherKey[i];

    // Obtain the following parts of the ExpandedKey, creating a word (4 bytes)
    // during each iteration
    for (int i = 16; i < 16 * (NR_ROUNDS + 1); i += 4) {
        // Store the current last word of the ExpandedKey
        char temp[4];
        for (int j = 0; j < 4; j++)
            temp[j] = ExpandedKey[(i - 4) + j];

        // If the current word is a multiple of the key length, then apply
        // a transformation
        if (i % 16 == 0) {
            RotByte(temp);

            SubByte(temp, sbox);
            temp[0] ^= rcon[i / 16];
        }

        // The next word of the ExpandedKey is equal to the bitwise EXOR
        // of the current last word and the word came 4 words before the
        // word that is currently computed
        for (int j = 0; j < 4; j++)
            ExpandedKey[i + j] = ExpandedKey[(i - 16) + j] ^ temp[j];
    }
}


__device__ void FinalRound(char *block, char *roundkey, char *sbox) {
    SubBytes(block, sbox);
    ShiftRows(block);
    AddRoundKey(block, roundkey);
}


__device__ void InvSubBytes(char *block, char *invSbox) {
    for (unsigned int i = 0; i < 16; i++) {
        block[i] = invSbox[(unsigned char) block[i]];
    }
}

__device__ void InvShiftRows(char *block) {
    // ROW 0: Untouched

    // ROW 1: Shift right by one
    char temp = block[13];
    block[13] = block[9];
    block[9] = block[5];
    block[5] = block[1];
    block[1] = temp;

    // ROW 2: Shift right by two
    temp = block[14];
    block[14] = block[6];
    block[6] = temp;
    temp = block[10];
    block[10] = block[2];
    block[2] = temp;

    // ROW 3: Shift right by three
    temp = block[3];
    block[3] = block[7];
    block[7] = block[11];
    block[11] = block[15];
    block[15] = temp;
}

__device__ void invMixColumns(char *block, char *mul2, char *mul3) {
    char u;
    char v;
    for (unsigned int col = 0; col < 4; col++) {
        u = mul2[(unsigned char) mul2[(unsigned char) (block[4 * col] ^ block[4 * col + 2])]];
        v = mul2[(unsigned char) mul2[(unsigned char) (block[4 * col + 1] ^ block[4 * col + 3])]];
        block[4 * col] = block[4 * col] ^ u;
        block[4 * col + 1] = block[4 * col + 1] ^ v;
        block[4 * col + 2] = block[4 * col + 2] ^ u;
        block[4 * col + 3] = block[4 * col + 3] ^ v;
    }


    mixColumns(block, mul2, mul3);
}


__device__ void InvRound(char *block, char *roundKey, char *invSbox, char *mul2, char *mul3) {
    InvShiftRows(block);
    InvSubBytes(block, invSbox);
    AddRoundKey(block, roundKey);
    invMixColumns(block, mul2, mul3);
}


__device__ void InvFinalRound(char *block, char *roundkey, char *invSbox) {
    InvShiftRows(block);
    InvSubBytes(block, invSbox);
    AddRoundKey(block, roundkey);
}


__global__ void AES(char *State, char *CipherKey, const unsigned int StateLength, char *grcon,
                    char *gsbox, char *gmul2, char *gmul3) {
    int index = (threadIdx.x + blockDim.x * blockIdx.x) * 16; // * 16 because every thread processes an entire block

    // Load the lookup tables into shared memory
    __shared__ char rcon[256];
    __shared__ char sbox[256];
    __shared__ char mul2[256];
    __shared__ char mul3[256];

    if (blockDim.x < 256) {
        if (threadIdx.x == 0) {
            for (int i = 0; i < 256; i++) {
                rcon[i] = grcon[i];
                sbox[i] = gsbox[i];
                mul2[i] = gmul2[i];
                mul3[i] = gmul3[i];
            }
        }
    } else {
        if (threadIdx.x < 256) {
            rcon[threadIdx.x] = grcon[threadIdx.x];
            sbox[threadIdx.x] = gsbox[threadIdx.x];
            mul2[threadIdx.x] = gmul2[threadIdx.x];
            mul3[threadIdx.x] = gmul3[threadIdx.x];
        }
    }
    __syncthreads();


    // Only a single thread from the thread block must calculate the ExpanedKey
    __shared__ char ExpandedKey[16 * (NR_ROUNDS + 1)];
    if (threadIdx.x == 0)
        KeyExpansion(CipherKey, ExpandedKey, rcon, sbox);

    // Load State into private memory (a state is only used by a single thread)
    char stateLocal[16];
    if (index + 16 <= StateLength) {
        for (int i = 0; i < 16; i++) {
            stateLocal[i] = State[index + i];
        }
    }

    // Synchronize the threads because thread 0 wrote to shared memory, and
    // the ExpanedKey will be accessed by each thread in the block.
    __syncthreads();

    // Each thread handles 16 bytes (a single block) of the State
    if (index + 16 <= StateLength) {
        AddRoundKey(stateLocal, ExpandedKey);
        for (int i = 1; i < NR_ROUNDS; i++)
            Round(stateLocal, ExpandedKey + 16 * i, sbox, mul2, mul3);
        FinalRound(stateLocal, ExpandedKey + 16 * NR_ROUNDS, sbox);
    }

    __syncthreads();

    // Write back the results to State
    if (index + 16 <= StateLength)
        for (int i = 0; i < 16; i++)
            State[index + i] = stateLocal[i];
}

__global__ void inv_AES(char *state, char *cipherKey, const unsigned int stateLength, char *grcon, char *gsbox,
                        char *ginvsbox, char *gmul2, char *gmul3) {
    int index = (threadIdx.x + blockDim.x * blockIdx.x) * 16; // * 16 because every thread processes an entire block

    // Load the lookup tables into shared memory
    __shared__ char rcon[256];
    __shared__ char sbox[256]; // required for key expansion
    __shared__ char invsbox[256];
    __shared__ char mul2[256];
    __shared__ char mul3[256];

    if (blockDim.x < 256) {
        if (threadIdx.x == 0) {
            for (int i = 0; i < 256; i++) {
                rcon[i] = grcon[i];
                sbox[i] = gsbox[i];
                invsbox[i] = ginvsbox[i];
                mul2[i] = gmul2[i];
                mul3[i] = gmul3[i];
            }
        }
    } else {
        if (threadIdx.x < 256) {
            rcon[threadIdx.x] = grcon[threadIdx.x];
            sbox[threadIdx.x] = gsbox[threadIdx.x];
            invsbox[threadIdx.x] = ginvsbox[threadIdx.x];
            mul2[threadIdx.x] = gmul2[threadIdx.x];
            mul3[threadIdx.x] = gmul3[threadIdx.x];
        }
    }
    __syncthreads();


    // Only a single thread from the thread block must calculate the ExpanedKey
    __shared__ char ExpandedKey[16 * (NR_ROUNDS + 1)];
    if (threadIdx.x == 0)
        KeyExpansion(cipherKey, ExpandedKey, rcon, sbox);

    // Load State into private memory (a state is only used by a single thread)
    char stateLocal[16];
    if (index + 16 <= stateLength) {
        for (int i = 0; i < 16; i++) {
            stateLocal[i] = state[index + i];
        }
    }

    // Synchronize the threads because thread 0 wrote to shared memory, and
    // the ExpanedKey will be accessed by each thread in the block.
    __syncthreads();

    // Each thread handles 16 bytes (a single block) of the State
    if (index + 16 <= stateLength) {
        AddRoundKey(stateLocal, ExpandedKey + 16 * NR_ROUNDS);
        for (int i = 1; i < NR_ROUNDS; i++) {
            InvRound(stateLocal, ExpandedKey + 16 * NR_ROUNDS - 16 * i, invsbox, mul2, mul3);
            // now run through the expanded key in reverse!
        }
        InvFinalRound(stateLocal, ExpandedKey, invsbox);
    }

    __syncthreads();
    // Write back the results to State
    if (index + 16 <= stateLength)
        for (int i = 0; i < 16; i++) {
            state[index + i] = stateLocal[i];
        }
}

__global__ void AES_CTR(char *State, char *CipherKey, const unsigned int StateLength, char *grcon, char *gsbox,
                        char *gmul2, char *gmul3, unsigned int counterinit) {
    int index = (threadIdx.x + blockDim.x * blockIdx.x) * 16; // Cada hilo procesa un bloque completo (16 bytes)

    // Cargar las tablas de búsqueda en memoria compartida
    __shared__ char rcon[256];
    __shared__ char sbox[256];
    __shared__ char mul2[256];
    __shared__ char mul3[256];

    if (blockDim.x < 256) {
        if (threadIdx.x == 0) {
            for (int i = 0; i < 256; i++) {
                rcon[i] = grcon[i];
                sbox[i] = gsbox[i];
                mul2[i] = gmul2[i];
                mul3[i] = gmul3[i];
            }
        }
    } else {
        if (threadIdx.x < 256) {
            rcon[threadIdx.x] = grcon[threadIdx.x];
            sbox[threadIdx.x] = gsbox[threadIdx.x];
            mul2[threadIdx.x] = gmul2[threadIdx.x];
            mul3[threadIdx.x] = gmul3[threadIdx.x];
        }
    }
    __syncthreads();

    // Solo un hilo del bloque de hilos debe calcular el ExpandedKey
    __shared__ char ExpandedKey[16 * (NR_ROUNDS + 1)];
    if (threadIdx.x == 0)
        KeyExpansion(CipherKey, ExpandedKey, rcon, sbox);

    // Sincronización de hilos después de la expansión de la clave
    __syncthreads();

    // Cargar el estado local y el contador
    char stateLocal[16];
    char counter[16];

    if (index + 16 <= StateLength) {
        for (int i = 0; i < 16; i++) {
            stateLocal[i] = State[index + i];
            counter[i] = 0;
        }
        // Inicializar la parte baja del contador
        unsigned int *counterPtr = (unsigned int *) counter;
        counterPtr[0] = counterinit;
        counterPtr[1] = index / 16;
    }

    // Cifrar el contador usando AES-ECB
    if (index + 16 <= StateLength) {
        AddRoundKey(counter, ExpandedKey);
        for (int i = 1; i < NR_ROUNDS; i++)
            Round(counter, ExpandedKey + 16 * i, sbox, mul2, mul3);
        FinalRound(counter, ExpandedKey + 16 * NR_ROUNDS, sbox);
    }

    __syncthreads();

    // Realizar XOR entre el contador cifrado y el texto plano para obtener el texto cifrado
    if (index + 16 <= StateLength) {
        for (int i = 0; i < 16; i++) {
            State[index + i] = stateLocal[i] ^ counter[i];
        }
    }
}

__global__ void AES_CCMP_1(char *Data, char *CipherKey, const unsigned int DataLength,
                           char *grcon, char *gsbox, char *gmul2, char *gmul3,
                           unsigned int nonce, unsigned char *AAD, unsigned int AADLength,
                           unsigned char *MIC) {
    int index = (threadIdx.x + blockDim.x * blockIdx.x) * 16;

    // Cargar tablas en memoria compartida
    __shared__ char rcon[256];
    __shared__ char sbox[256];
    __shared__ char mul2[256];
    __shared__ char mul3[256];

    if (blockDim.x < 256) {
        if (threadIdx.x == 0) {
            for (int i = 0; i < 256; i++) {
                rcon[i] = grcon[i];
                sbox[i] = gsbox[i];
                mul2[i] = gmul2[i];
                mul3[i] = gmul3[i];
            }
        }
    } else {
        if (threadIdx.x < 256) {
            rcon[threadIdx.x] = grcon[threadIdx.x];
            sbox[threadIdx.x] = gsbox[threadIdx.x];
            mul2[threadIdx.x] = gmul2[threadIdx.x];
            mul3[threadIdx.x] = gmul3[threadIdx.x];
        }
    }
    __syncthreads();

    // Expandir la clave
    __shared__ char ExpandedKey[16 * (NR_ROUNDS + 1)];
    if (threadIdx.x == 0)
        KeyExpansion(CipherKey, ExpandedKey, rcon, sbox);
    __syncthreads();

    // Inicializar contador y CBC-MAC
    char counter[16] = {0};
    char cbcMac[16] = {0};

    // Configurar contador inicial
    *(unsigned int *) (&counter[0]) = nonce;
    *(unsigned int *) (&counter[4]) = index / 16;

    // Procesar AAD para CBC-MAC
    if (threadIdx.x == 0) {
        for (unsigned int i = 0; i < AADLength; i += 16) {
            for (int j = 0; j < 16 && i + j < AADLength; j++)
                cbcMac[j] ^= AAD[i + j];

            AddRoundKey(cbcMac, ExpandedKey);
            for (int r = 1; r < NR_ROUNDS; r++)
                Round(cbcMac, ExpandedKey + 16 * r, sbox, mul2, mul3);
            FinalRound(cbcMac, ExpandedKey + 16 * NR_ROUNDS, sbox);
        }
    }
    __syncthreads();

    // Procesar los datos
    if (index + 16 <= DataLength) {
        char dataBlock[16];
        for (int i = 0; i < 16; i++)
            dataBlock[i] = Data[index + i];

        // Encriptar el contador
        char encryptedCounter[16];
        for (int i = 0; i < 16; i++)
            encryptedCounter[i] = counter[i];

        AddRoundKey(encryptedCounter, ExpandedKey);
        for (int r = 1; r < NR_ROUNDS; r++)
            Round(encryptedCounter, ExpandedKey + 16 * r, sbox, mul2, mul3);
        FinalRound(encryptedCounter, ExpandedKey + 16 * NR_ROUNDS, sbox);

        // XOR con los datos para encriptar/desencriptar
        for (int i = 0; i < 16; i++)
            Data[index + i] ^= encryptedCounter[i];

        // Actualizar CBC-MAC
        for (int i = 0; i < 16; i++)
            cbcMac[i] ^= dataBlock[i];

        AddRoundKey(cbcMac, ExpandedKey);
        for (int r = 1; r < NR_ROUNDS; r++)
            Round(cbcMac, ExpandedKey + 16 * r, sbox, mul2, mul3);
        FinalRound(cbcMac, ExpandedKey + 16 * NR_ROUNDS, sbox);
    }

    // El último hilo calcula el MIC final
    if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) {
        for (int i = 0; i < 16; i++)
            MIC[i] = cbcMac[i]; // Guardar el MIC calculado
    }
}

__global__ void AES_CCMP(char *State, char *CipherKey, const unsigned int StateLength, char *grcon, char *gsbox,
                         char *gmul2, char *gmul3, unsigned int nonce, unsigned int counter,
                         unsigned char *AAD, unsigned int AADLength, unsigned char *MIC) {
    int index = (threadIdx.x + blockDim.x * blockIdx.x) * 16;

    // Cargar las tablas de búsqueda en memoria compartida
    __shared__ char rcon[256];
    __shared__ char sbox[256];
    __shared__ char mul2[256];
    __shared__ char mul3[256];

    if (blockDim.x < 256) {
        if (threadIdx.x == 0) {
            for (int i = 0; i < 256; i++) {
                rcon[i] = grcon[i];
                sbox[i] = gsbox[i];
                mul2[i] = gmul2[i];
                mul3[i] = gmul3[i];
            }
        }
    } else {
        if (threadIdx.x < 256) {
            rcon[threadIdx.x] = grcon[threadIdx.x];
            sbox[threadIdx.x] = gsbox[threadIdx.x];
            mul2[threadIdx.x] = gmul2[threadIdx.x];
            mul3[threadIdx.x] = gmul3[threadIdx.x];
        }
    }
    __syncthreads();

    // Expansión de clave (solo un hilo por bloque)
    __shared__ char ExpandedKey[16 * (NR_ROUNDS + 1)];
    if (threadIdx.x == 0)
        KeyExpansion(CipherKey, ExpandedKey, rcon, sbox);

    __syncthreads();

    // Inicialización de variables locales
    char stateLocal[16];
    char counterBlock[16];
    char cbcMac[16];

    // Inicialización del bloque contador y CBC-MAC
    if (index == 0) {
        // Inicializar el bloque contador
        memset(counterBlock, 0, 16);
        *(unsigned int *) &counterBlock[0] = nonce;
        *(unsigned int *) &counterBlock[12] = counter;

        // Inicializar CBC-MAC
        memset(cbcMac, 0, 16);
        // Procesar AAD (Additional Authentication Data)
        for (unsigned int i = 0; i < AADLength; i += 16) {
            for (int j = 0; j < 16 && i + j < AADLength; j++) {
                cbcMac[j] ^= AAD[i + j];
            }
            // Cifrar el bloque CBC-MAC
            AddRoundKey(cbcMac, ExpandedKey);
            for (int r = 1; r < NR_ROUNDS; r++)
                Round(cbcMac, ExpandedKey + 16 * r, sbox, mul2, mul3);
            FinalRound(cbcMac, ExpandedKey + 16 * NR_ROUNDS, sbox);
        }
    }

    __syncthreads();

    // Procesar el State (datos a cifrar)
    if (index + 16 <= StateLength) {
        // Cargar el bloque de State
        for (int i = 0; i < 16; i++) {
            stateLocal[i] = State[index + i];
        }

        // Actualizar el contador para este bloque
        *(unsigned int *) &counterBlock[12] = counter + (index / 16);

        // Cifrar el bloque contador
        char encryptedCounter[16];
        memcpy(encryptedCounter, counterBlock, 16);
        AddRoundKey(encryptedCounter, ExpandedKey);
        for (int r = 1; r < NR_ROUNDS; r++)
            Round(encryptedCounter, ExpandedKey + 16 * r, sbox, mul2, mul3);
        FinalRound(encryptedCounter, ExpandedKey + 16 * NR_ROUNDS, sbox);

        // XOR con el State para cifrar
        for (int i = 0; i < 16; i++) {
            stateLocal[i] ^= encryptedCounter[i];
        }

        // Actualizar CBC-MAC
        for (int i = 0; i < 16; i++) {
            cbcMac[i] ^= stateLocal[i];
        }
        AddRoundKey(cbcMac, ExpandedKey);
        for (int r = 1; r < NR_ROUNDS; r++)
            Round(cbcMac, ExpandedKey + 16 * r, sbox, mul2, mul3);
        FinalRound(cbcMac, ExpandedKey + 16 * NR_ROUNDS, sbox);

        // Escribir el resultado cifrado de vuelta a State
        for (int i = 0; i < 16; i++) {
            State[index + i] = stateLocal[i];
        }
    }

    __syncthreads();

    // El último hilo escribe el MIC (Message Integrity Code)
    if (index + 16 > StateLength && index < StateLength + 16) {
        for (int i = 0; i < 16; i++) {
            MIC[i] = cbcMac[i];
        }
    }
}

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

__global__ void Calculate_MIC(char *State, const unsigned int StateLength,
                              char *CipherKey, char *grcon, char *gsbox,
                              char *gmul2, char *gmul3,
                              unsigned char *AAD, unsigned int AADLength,
                              unsigned char *MIC) {
    // Este kernel se ejecutará con un solo hilo o bloque
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        char cbcMac[16];
        char ExpandedKey[16 * (NR_ROUNDS + 1)];

        // Cargar tablas de búsqueda
        char rcon[256], sbox[256], mul2[256], mul3[256];
        for (int i = 0; i < 256; i++) {
            rcon[i] = grcon[i];
            sbox[i] = gsbox[i];
            mul2[i] = gmul2[i];
            mul3[i] = gmul3[i];
        }

        // Expansión de clave
        KeyExpansion(CipherKey, ExpandedKey, rcon, sbox);

        // Inicializar CBC-MAC
        memset(cbcMac, 0, 16);

        // Procesar AAD
        for (unsigned int i = 0; i < AADLength; i += 16) {
            for (int j = 0; j < 16 && i + j < AADLength; j++) {
                cbcMac[j] ^= AAD[i + j];
            }
            AddRoundKey(cbcMac, ExpandedKey);
            for (int r = 1; r < NR_ROUNDS; r++)
                Round(cbcMac, ExpandedKey + 16 * r, sbox, mul2, mul3);
            FinalRound(cbcMac, ExpandedKey + 16 * NR_ROUNDS, sbox);
        }

        // Procesar State
        for (unsigned int i = 0; i < StateLength; i += 16) {
            for (int j = 0; j < 16 && i + j < StateLength; j++) {
                cbcMac[j] ^= State[i + j];
            }
            AddRoundKey(cbcMac, ExpandedKey);
            for (int r = 1; r < NR_ROUNDS; r++)
                Round(cbcMac, ExpandedKey + 16 * r, sbox, mul2, mul3);
            FinalRound(cbcMac, ExpandedKey + 16 * NR_ROUNDS, sbox);
        }

        // Escribir MIC final
        for (int i = 0; i < 16; i++) {
            MIC[i] = cbcMac[i];
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
