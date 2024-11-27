
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
