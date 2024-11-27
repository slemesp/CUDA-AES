
        #define LUT_IN_SHARED
        #define NR_ROUNDS 10

#ifndef LUT_IN_SHARED

__constant__ char rcon[256];
__constant__ char sbox[256];
__constant__ char invSbox[256];
__constant__ char mul2[256];
__constant__ char mul3[256];

#endif

// analogous to subBytes, but with the inverse of the original sbox
#ifndef LUT_IN_SHARED
__device__ void InvSubBytes(char* block){
#else
__device__ void InvSubBytes(char* block, char* invSbox){
#endif
    for(unsigned int i = 0; i < 16; i++){
        block[i] = invSbox[(unsigned char) block[i]];
    }
}
__device__ void InvShiftRows(char* block)
{
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

#ifdef TEST_INVSHIFTROWS

__global__ void InvShiftRowsTest(char* message, const unsigned int length)
{
    int idx = (threadIdx.x + blockDim.x * blockIdx.x) * 16; // * 16 because every thread processes an entire block

    if (idx + 16 <= length)
        InvShiftRows(message + idx);
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

// LUTs for the tests are defined in mixColumns. This file should be compiled first, as invMixColumns is dependent on mixColumns

#ifndef LUT_IN_SHARED
__device__ void invMixColumns(char* block){
#else
__device__ void invMixColumns(char* block, char* mul2, char* mul3){
#endif
    char u;
    char v;
    for (unsigned int col = 0; col < 4; col++){
        u = mul2[(unsigned char) mul2[(unsigned char) (block[4*col] ^ block[4*col + 2])]];
        v = mul2[(unsigned char) mul2[(unsigned char) (block[4*col + 1] ^ block[4*col + 3])]];
        block[4*col] = block[4*col] ^ u;
        block[4*col + 1] = block[4*col + 1] ^ v;
        block[4*col + 2] = block[4*col + 2] ^ u;
        block[4*col + 3] = block[4*col + 3] ^ v;
    }

    #ifndef LUT_IN_SHARED
    mixColumns(block);
    #else
    mixColumns(block, mul2, mul3);
    #endif
}


#ifdef TEST_INVMIXCOLUMNS

__global__ void invMixColumnsTest(char* message, const unsigned int length){
    int idx = (threadIdx.x + blockDim.x * blockIdx.x) * 16;

    if (idx + 16 <= length){
        invMixColumns(message + idx);
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

#ifndef LUT_IN_SHARED
__device__ void InvRound(char* block, char* roundKey){
    InvShiftRows(block);
    InvSubBytes(block);
    AddRoundKey(block, roundKey);
    invMixColumns(block);
}
#else
__device__ void InvRound(char* block, char* roundKey, char* invSbox, char* mul2, char* mul3){
    InvShiftRows(block);
    InvSubBytes(block, invSbox);
    AddRoundKey(block, roundKey);
    invMixColumns(block, mul2, mul3);
}
#endif

#ifdef TEST_INVROUND

__global__ void InvRoundTest(char* block, char* roundkey)
{
    InvRound(block, roundkey);
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

#ifndef LUT_IN_SHARED
__device__ void InvFinalRound(char* block, char* roundkey)
{
    InvShiftRows(block);
    InvSubBytes(block);
    AddRoundKey(block, roundkey);
}
#else
__device__ void InvFinalRound(char* block, char* roundkey, char* invSbox)
{
    InvShiftRows(block);
    InvSubBytes(block, invSbox);
    AddRoundKey(block, roundkey);
}
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

/*
    The stucture of encryption and decryption is exactly the same, so the fastest 
    encryption method is used to implement the decryption algorithm. The fastest
    method was found to be storing the state in private memory, and the LUTs in shared
    memory, which avoids the use of constant memory

    inputs (different from AES):
        - mem_initialized: indicates whether the LUTs were already loaded 
*/

__global__ void inv_AES(char* state, char* cipherKey, const unsigned int stateLength, char* grcon, char* gsbox, char* ginvsbox, char* gmul2, char* gmul3){
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
    if(index + 16 <= stateLength){
        for(int i = 0; i < 16; i++){
            stateLocal[i] = state[index + i];
        }
    }

    // Synchronize the threads because thread 0 wrote to shared memory, and
    // the ExpanedKey will be accessed by each thread in the block.
    __syncthreads();

    // Each thread handles 16 bytes (a single block) of the State
    if (index + 16 <= stateLength)
    {
        AddRoundKey(stateLocal, ExpandedKey + 16 * NR_ROUNDS);
        for (int i = 1; i < NR_ROUNDS; i++){
            InvRound(stateLocal, ExpandedKey + 16*NR_ROUNDS - 16*i, invsbox, mul2, mul3); // now run through the expanded key in reverse!
        }
        InvFinalRound(stateLocal, ExpandedKey, invsbox);
    }

    __syncthreads();
    // Write back the results to State
    if (index + 16 <= stateLength)
        for (int i = 0; i < 16; i++){
            state[index + i] = stateLocal[i];
        }

}

