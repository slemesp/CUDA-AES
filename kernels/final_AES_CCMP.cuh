
        #define AES_PRIVATESTATE_SHAREDLUT
        #define LUT_IN_SHARED
        #define NR_ROUNDS 10

#ifndef LUT_IN_SHARED

__constant__ char rcon[256];
__constant__ char sbox[256];
__constant__ char invSbox[256];
__constant__ char mul2[256];
__constant__ char mul3[256];

#endif

#ifdef TEST_SUBBYTES

__constant__ char sbox[256];

#endif

#ifndef LUT_IN_SHARED
__device__ void SubBytes(char* block){
#else
__device__ void SubBytes(char* block, char* sbox){
#endif
    for(unsigned int i = 0; i < 16; i++){
        block[i] = sbox[(unsigned char) block[i]];
    }
}

#ifdef TEST_SUBBYTES

__global__ void SubByteTest(char* message, const unsigned int length){
    int idx = (threadIdx.x + blockDim.x * blockIdx.x) * 16;
    if (idx + 16 <= length){
        SubBytes(message + idx);
    }
}

#endif

/*
   ShiftRows operation:
   Shift the bytes within a block as follows:
   b0  b4  b8  b12 -> b0  b4  b8  b12 (no shift)
   b1  b5  b9  b13 -> b5  b9  b13 b1  (shift left by one)
   b2  b6  b10 b14 -> b10 b14 b2  b6  (shift left by two)
   b3  b7  b11 b15 -> b15 b3  b7  b11 (shift left by three)

   InOut:
    - block: char array of length 16
*/

__device__ void ShiftRows(char* block)
{
    // ROW 0: Untouched

    // ROW 1: Shift left by one
    char temp = block[1];
    block[1]  = block[5];
    block[5]  = block[9];
    block[9]  = block[13];
    block[13] = temp;

    // ROW 2: Shift left by two
    temp = block[2];
    block[2]  = block[10];
    block[10] = temp;
    temp = block[6];
    block[6]  = block[14];
    block[14] = temp;

    // ROW 3: Shift left by three
    temp = block[15];
    block[15] = block[11];
    block[11] = block[7];
    block[7]  = block[3];
    block[3]  = temp;
}

#ifdef TEST_SHIFTROWS

__global__ void ShiftRowsTest(char* message, const unsigned int length)
{
    int idx = (threadIdx.x + blockDim.x * blockIdx.x) * 16; // * 16 because every thread processes an entire block

    if (idx + 16 <= length)
        ShiftRows(message + idx);
}

#endif

#ifdef TEST_MIXCOLUMNS

__constant__ char mul2[256];
__constant__ char mul3[256];

# endif
/*
    xtime:
        There is no simple operation on the byte level that corresponds to a
        finite field multiplication. A multiplication by x, however, can be implemented
        as a byte level operation. Thus, multiplication by higher powers of x are
        repetitions of xtime.

        input:
            poly: byte representing a polynomial
        output:
            out: poly multiplied by x
*/

__device__ char xtime(unsigned char poly){
    char out = poly << 1;
    if (!(out & 0x80)){     // polynomial not yet in reduced form
        out ^= 0x1b;
    }
    return out;
}

/* Multiply input polynomial by x + 1
__device__ char mul3(char poly){
    return poly ^ xtime(poly);
}
*/

#ifndef LUT_IN_SHARED
__device__ void mixColumns(char* block){
#else
__device__ void mixColumns(char* block, char* mul2, char* mul3){
#endif

    char temp[16];
    for (unsigned int col = 0; col < 4; col++){
        temp[4*col] = mul2[(unsigned char) block[4*col]] ^ mul3[(unsigned char) block[4*col + 1]] ^ block[4*col + 2] ^ block[4*col + 3];
        temp[4*col + 1] = block[4*col] ^ mul2[(unsigned char) block[4*col + 1]] ^ mul3[(unsigned char) block[4*col + 2]] ^ block[4*col + 3];
        temp[4*col + 2] = block[4*col] ^ block[4*col + 1] ^ mul2[(unsigned char) block[4*col + 2]] ^ mul3[(unsigned char) block[4*col + 3]];
        temp[4*col + 3] = mul3[(unsigned char) block[4*col]] ^ block[4*col + 1] ^ block[4*col + 2] ^ mul2[(unsigned char) block[4*col + 3]];

    }

    for (unsigned int i = 0; i < 16; i++){
        block[i] = temp[i];
    }
}

#ifdef TEST_MIXCOLUMNS

__global__ void xtime_test(char* poly){
    if(threadIdx.x == 0){
        poly[blockIdx.x] = xtime(poly[blockIdx.x]);
    }
}

__global__ void mixColumnsTest(char* message, const unsigned int length){
    int idx = (threadIdx.x + blockDim.x * blockIdx.x) * 16;

    if (idx + 16 <= length){
        mixColumns(message + idx);
    }
}

#endif

/*
   AddRoundKey operation:
   Apply RoundKey to the block by performing a bitwise EXOR between the bytes
   from the block and the corresponding bytes from the RoundKey.

   Input:
    - RoundKey: char array of length 16
   InOut:
    - block: char array of length 16
*/

__device__ void AddRoundKey(char* block, char* RoundKey)
{
    block[0]  ^= RoundKey[0];
    block[1]  ^= RoundKey[1];
    block[2]  ^= RoundKey[2];
    block[3]  ^= RoundKey[3];
    block[4]  ^= RoundKey[4];
    block[5]  ^= RoundKey[5];
    block[6]  ^= RoundKey[6];
    block[7]  ^= RoundKey[7];
    block[8]  ^= RoundKey[8];
    block[9]  ^= RoundKey[9];
    block[10] ^= RoundKey[10];
    block[11] ^= RoundKey[11];
    block[12] ^= RoundKey[12];
    block[13] ^= RoundKey[13];
    block[14] ^= RoundKey[14];
    block[15] ^= RoundKey[15];
}

#ifdef TEST_ROUNDKEY

__global__ void AddRoundKeyTest(char* message, char* roundkey, const unsigned int length)
{
    int idx = (threadIdx.x + blockDim.x * blockIdx.x) * 16; // * 16 because every thread processes an entire block
    if (idx + 16 <= length)
        AddRoundKey(message + idx, roundkey);
}

#endif

/*
   Round operation:
   Perform the following operations on the input block:
    1. SubBytes
    2. ShiftRows
    3. MixColumns
    4. AddRoundKey
   This operation as a whole is a single round operation from the AES
   algorithm. The RoundKey used is one block of the ExpandedKey.

   Input:
    - roundkey: char array of length 16

   InOut:
    - block: char array of length 16
*/

#ifndef LUT_IN_SHARED
__device__ void Round(char* block, char* roundkey)
{
    SubBytes(block);
    ShiftRows(block);
    mixColumns(block);
    AddRoundKey(block, roundkey);
}
#else
__device__ void Round(char* block, char* roundkey, char* sbox, char* mul2, char* mul3)
{
    SubBytes(block, sbox);
    ShiftRows(block);
    mixColumns(block, mul2, mul3);
    AddRoundKey(block, roundkey);
}
#endif

#ifdef TEST_ROUND

__global__ void RoundTest(char* block, char* roundkey)
{
    Round(block, roundkey);
}

#endif

#ifdef TEST_KEYEXPANSION

#define NR_ROUNDS 10

__constant__ char rcon[256];
__constant__ char sbox[256];

#endif

/*
   RotByte:
   Rotate the bytes within a 4-byte array as follows:
   a b c d -> b c d a

   InOut:
    - array: char array of length 4
*/

__device__ void RotByte(char* array)
{
    char temp = array[0];
    array[0] = array[1];
    array[1] = array[2];
    array[2] = array[3];
    array[3] = temp;
}

/*
   SubByte:
   Apply the sbox lookup table to each byte.

   InOut:
    - array: char array of length 4
*/

#ifndef LUT_IN_SHARED
__device__ void SubByte(char* array)
#else
__device__ void SubByte(char* array, char* sbox)
#endif
{
    // Need to convert to unsigned char for correct indexing
    array[0] = sbox[(unsigned char) array[0]];
    array[1] = sbox[(unsigned char) array[1]];
    array[2] = sbox[(unsigned char) array[2]];
    array[3] = sbox[(unsigned char) array[3]];
}

/*
   KeyExpansion:
   Create the expanded key from the Cipher Key. The expanded key has
   16 * (NR_ROUNDS + 1) bytes, and is chopped into pieces of 16 bytes to obtain
   the RoundKey for each round.

   Input:
    - CipherKey: char array of length 4

   Output:
    - ExpandedKey: char array of length 16 * (NR_ROUNDS + 1)
*/

#ifndef LUT_IN_SHARED
__device__ void KeyExpansion(char* CipherKey, char* ExpandedKey)
#else
__device__ void KeyExpansion(char* CipherKey, char* ExpandedKey, char* rcon, char* sbox)
#endif
{
    // First part of the expanded key is equal to the Cipher key.
    for (int i = 0; i < 16; i++)
        ExpandedKey[i] = CipherKey[i];

    // Obtain the following parts of the ExpandedKey, creating a word (4 bytes)
    // during each iteration
    for (int i = 16; i < 16 * (NR_ROUNDS + 1); i += 4)
    {
        // Store the current last word of the ExpandedKey
        char temp[4];
        for (int j = 0; j < 4; j++)
            temp[j] = ExpandedKey[(i - 4) + j];

        // If the current word is a multiple of the key length, then apply
        // a transformation
        if (i % 16 == 0)
        {
            RotByte(temp);
#ifndef LUT_IN_SHARED
            SubByte(temp);
#else
            SubByte(temp, sbox);
#endif
            temp[0] ^= rcon[i / 16];
        }

        // The next word of the ExpandedKey is equal to the bitwise EXOR
        // of the current last word and the word came 4 words before the
        // word that is currently computed
        for (int j = 0; j < 4; j++)
            ExpandedKey[i + j] = ExpandedKey[(i - 16) + j] ^ temp[j];
    }

}

#ifdef TEST_KEYEXPANSION

__global__ void KeyExpansionTest(char* cipherkey, char* expandedkey)
{
    KeyExpansion(cipherkey, expandedkey);
}

#endif

#ifdef TEST_FINALROUND // todo: delete or write test, this is never used in the current codebase

#define NR_ROUNDS 10

__constant__ char rcon[256];
__constant__ char sbox[256];
__constant__ char mul2[256];
__constant__ char mul3[256];

#endif

/*
   FinalRound operation:
   Perform the following operations on the input block:
    1. ByteSub
    2. ShiftRows
    3. AddRoundKey
   This operation as a whole is the final round operation from the AES
   algorithm. The RoundKey used is one block of the ExpandedKey.

   Input:
    - roundkey: char array of length 16

   InOut:
    - block: char array of length 16
*/

#ifndef LUT_IN_SHARED
__device__ void FinalRound(char* block, char* roundkey)
{
    SubBytes(block);
    ShiftRows(block);
    AddRoundKey(block, roundkey);
}
#else
__device__ void FinalRound(char* block, char* roundkey, char* sbox)
{
    SubBytes(block, sbox);
    ShiftRows(block);
    AddRoundKey(block, roundkey);
}
#endif

#ifdef FINALTEST_ROUND

__global__ void FinalRoundTest(char* block, char* roundkey)
{
    FinalRound(block, roundkey);
}

#endif

#ifndef GHASH_CUH
#define GHASH_CUH

__device__ void GHash(char* H, char* X, unsigned int X_len, char* Y) {
    char Z[16] = {0};
    for (unsigned int i = 0; i < X_len; i += 16) {
        for (int j = 0; j < 16; j++) {
            Z[j] ^= X[i + j];
        }
        // Galois field multiplication
        char V[16];
        for (int j = 0; j < 16; j++) V[j] = H[j];
        for (int j = 0; j < 128; j++) {
            if (Z[j / 8] & (1 << (7 - (j % 8)))) {
                for (int k = 0; k < 16; k++) Y[k] ^= V[k];
            }
            bool lsb = V[15] & 1;
            for (int k = 15; k > 0; k--) {
                V[k] = (V[k] >> 1) | ((V[k-1] & 1) << 7);
            }
            V[0] >>= 1;
            if (lsb) V[0] ^= 0xE1;
        }
    }
}

#endif // GHASH_CUH
#ifndef CTR_CUH
#define CTR_CUH
__device__ void AES_private_sharedlut(char* State, char* CipherKey, const unsigned int StateLength, char* grcon, char* gsbox, char* gmul2, char* gmul3)
{
    // Load the lookup tables into local memory
    char rcon[256];
    char sbox[256];
    char mul2[256];
    char mul3[256];

    for (int i = 0; i < 256; i++) {
        rcon[i] = grcon[i];
        sbox[i] = gsbox[i];
        mul2[i] = gmul2[i];
        mul3[i] = gmul3[i];
    }

    // Calculate the ExpandedKey
    char ExpandedKey[16 * (NR_ROUNDS + 1)];
    KeyExpansion(CipherKey, ExpandedKey, rcon, sbox);

    // Process the State
    char stateLocal[16];
    for(int i = 0; i < 16; i++){
        stateLocal[i] = State[i];
    }

    AddRoundKey(stateLocal, ExpandedKey);
    for (int i = 1; i < NR_ROUNDS; i++)
        Round(stateLocal, ExpandedKey + 16 * i, sbox, mul2, mul3);
    FinalRound(stateLocal, ExpandedKey + 16 * NR_ROUNDS, sbox);

    // Write back the results to State
    for (int i = 0; i < 16; i++)
        State[i] = stateLocal[i];
}

__device__ void CTR_mode(char* input, char* output, char* key, char* nonce, unsigned int len,
                         char* grcon, char* gsbox, char* gmul2, char* gmul3) {
    char counter[16];
    for (int i = 0; i < 12; i++) counter[i] = nonce[i]; // Copiar nonce en el contador
    *(unsigned int*)(counter + 12) = 1; // Inicializar el contador

    for (unsigned int i = 0; i < len; i += 16) {
        char encrypted_counter[16];

        // Llama a AES_private_sharedlut con los parámetros correctos
        AES_private_sharedlut(output, key, len, grcon, gsbox, gmul2, gmul3); // Cifrado del contador

        for (int j = 0; j < 16 && (i + j) < len; j++) {
            output[i + j] = input[i + j] ^ encrypted_counter[j]; // XOR con el texto plano
        }

        // Incrementar el contador
        for (int j = sizeof(counter) - 1; j >= sizeof(counter) - sizeof(unsigned int); j--) {
            if (++counter[j] != 0) break;
        }
    }
}

#endif // CTR_CUH
#ifndef GCMP_CUH
#define GCMP_CUH

// Función para cifrar usando GCMP
__global__ void GCMP_encrypt(char* plaintext, char* ciphertext, char* key, char* nonce,
                             char* auth_data, unsigned int plaintext_len,
                             unsigned int auth_data_len, char* tag,
                             char* grcon, char* gsbox, char* gmul2, char* gmul3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 16 < plaintext_len) {
        // Generar H (subclave hash)
        char H[16];
        AES_private_sharedlut(ciphertext, key, plaintext_len, grcon, gsbox, gmul2, gmul3); // Cifrado del nonce

        // Incrementar los últimos 32 bits del nonce para crear J0
        char J0[16];
        for (int i = 0; i < 12; i++) J0[i] = nonce[i];
        *(unsigned int*)(J0 + 12) = 1;

        // Cifrar el texto plano usando el modo CTR
        CTR_mode(plaintext + idx * 16, ciphertext + idx * 16, key, J0, 16, grcon, gsbox, gmul2, gmul3);

        // Calcular la etiqueta de autenticación
        char ghash_in[32];
        for (int i = 0; i < 16; i++) {
            ghash_in[i] = auth_data[idx * 16 + i];
            ghash_in[i + 16] = ciphertext[idx * 16 + i];
        }
        GHash(H, ghash_in, 32, tag);

        // XOR la etiqueta con E(K, J0)
        char E_J0[16];
        AES_private_sharedlut(ciphertext, key, plaintext_len, grcon, gsbox, gmul2, gmul3); // Cifrado del nonce
        for (int i = 0; i < 16; i++) tag[i] ^= E_J0[i];
    }
}

// Función para descifrar usando GCMP
__global__ void GCMP_decrypt(char* ciphertext, char* plaintext, char* key, char* nonce,
                             char* auth_data, unsigned int ciphertext_len,
                             unsigned int auth_data_len, char* tag,
                             char* grcon, char* gsbox, char* gmul2, char* gmul3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 16 < ciphertext_len) {
        // Generar H (subclave hash)
        char H[16];
        AES_private_sharedlut(ciphertext, key, ciphertext_len, grcon, gsbox, gmul2, gmul3); // Cifrado del nonce

        // Incrementar los últimos 32 bits del nonce para crear J0
        char J0[16];
        for (int i = 0; i < 12; i++) J0[i] = nonce[i];
        *(unsigned int*)(J0 + 12) = 1;

        // Calcular la etiqueta de autenticación
        char computed_tag[16];
        char ghash_in[32];
        for (int i = 0; i < 16; i++) {
            ghash_in[i] = auth_data[idx * 16 + i];
            ghash_in[i + 16] = ciphertext[idx * 16 + i];
        }
        GHash(H, ghash_in, 32, computed_tag);

        // XOR la etiqueta calculada con E(K, J0)
        char E_J0[16];
        AES_private_sharedlut(ciphertext, key, ciphertext_len, grcon, gsbox, gmul2, gmul3); // Cifrado del nonce
        for (int i = 0; i < 16; i++) computed_tag[i] ^= E_J0[i];

        // Verificar la etiqueta
        bool tag_valid = true;
        for (int i = 0; i < 16; i++) {
            if (computed_tag[i] != tag[i]) {
                tag_valid = false;
                break;
            }
        }

        // Si la etiqueta es válida, descifrar el texto cifrado
        if (tag_valid) {
            CTR_mode(ciphertext + idx * 16, plaintext + idx * 16, key, J0,
                     16, grcon, gsbox ,gmul2 ,gmul3);
        }
    }
}

#endif // GCMP_CUH
