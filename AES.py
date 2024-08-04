"""
This file contains the best performing AES encryption implementation and corresponding
decryption implementation, and the helper functions to easily encrypt and decrypt files
using this class by running this script. 
"""

import numpy as np
import pycuda.driver as cuda
from pycuda import autoinit
from pycuda.compiler import SourceModule

from utils.plot import plot_interactive_image


# from utils.KeyManager import KeyManager


# AES encryption and decryption class
class AES:
    """
    This class contains the best performing AES encryption implementation and corresponding
    """

    def __init__(self):
        self.dtype = np.uint8

        self.module_encrypt = None
        self.get_source_module_encrypt()
        self.get_source_module_decrypt()

        self.sbox = np.array([
            0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
            0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
            0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
            0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
            0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
            0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
            0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
            0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
            0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
            0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
            0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
            0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
            0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
            0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
            0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
        ], dtype=self.dtype)

        self.invSbox = np.array([
            0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
            0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
            0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
            0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
            0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
            0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
            0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
            0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
            0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
            0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
            0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
            0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
            0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
            0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
            0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
            0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
        ], dtype=self.dtype)

        self.rcon = np.array([
            0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a,
            0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39,
            0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a,
            0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8,
            0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef,
            0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc,
            0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b,
            0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3,
            0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94,
            0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20,
            0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35,
            0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f,
            0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04,
            0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63,
            0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd,
            0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d
        ], dtype=self.dtype)

        self.mul2 = np.array([
            0x00, 0x02, 0x04, 0x06, 0x08, 0x0a, 0x0c, 0x0e, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e,
            0x20, 0x22, 0x24, 0x26, 0x28, 0x2a, 0x2c, 0x2e, 0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3e,
            0x40, 0x42, 0x44, 0x46, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 0x52, 0x54, 0x56, 0x58, 0x5a, 0x5c, 0x5e,
            0x60, 0x62, 0x64, 0x66, 0x68, 0x6a, 0x6c, 0x6e, 0x70, 0x72, 0x74, 0x76, 0x78, 0x7a, 0x7c, 0x7e,
            0x80, 0x82, 0x84, 0x86, 0x88, 0x8a, 0x8c, 0x8e, 0x90, 0x92, 0x94, 0x96, 0x98, 0x9a, 0x9c, 0x9e,
            0xa0, 0xa2, 0xa4, 0xa6, 0xa8, 0xaa, 0xac, 0xae, 0xb0, 0xb2, 0xb4, 0xb6, 0xb8, 0xba, 0xbc, 0xbe,
            0xc0, 0xc2, 0xc4, 0xc6, 0xc8, 0xca, 0xcc, 0xce, 0xd0, 0xd2, 0xd4, 0xd6, 0xd8, 0xda, 0xdc, 0xde,
            0xe0, 0xe2, 0xe4, 0xe6, 0xe8, 0xea, 0xec, 0xee, 0xf0, 0xf2, 0xf4, 0xf6, 0xf8, 0xfa, 0xfc, 0xfe,
            0x1b, 0x19, 0x1f, 0x1d, 0x13, 0x11, 0x17, 0x15, 0x0b, 0x09, 0x0f, 0x0d, 0x03, 0x01, 0x07, 0x05,
            0x3b, 0x39, 0x3f, 0x3d, 0x33, 0x31, 0x37, 0x35, 0x2b, 0x29, 0x2f, 0x2d, 0x23, 0x21, 0x27, 0x25,
            0x5b, 0x59, 0x5f, 0x5d, 0x53, 0x51, 0x57, 0x55, 0x4b, 0x49, 0x4f, 0x4d, 0x43, 0x41, 0x47, 0x45,
            0x7b, 0x79, 0x7f, 0x7d, 0x73, 0x71, 0x77, 0x75, 0x6b, 0x69, 0x6f, 0x6d, 0x63, 0x61, 0x67, 0x65,
            0x9b, 0x99, 0x9f, 0x9d, 0x93, 0x91, 0x97, 0x95, 0x8b, 0x89, 0x8f, 0x8d, 0x83, 0x81, 0x87, 0x85,
            0xbb, 0xb9, 0xbf, 0xbd, 0xb3, 0xb1, 0xb7, 0xb5, 0xab, 0xa9, 0xaf, 0xad, 0xa3, 0xa1, 0xa7, 0xa5,
            0xdb, 0xd9, 0xdf, 0xdd, 0xd3, 0xd1, 0xd7, 0xd5, 0xcb, 0xc9, 0xcf, 0xcd, 0xc3, 0xc1, 0xc7, 0xc5,
            0xfb, 0xf9, 0xff, 0xfd, 0xf3, 0xf1, 0xf7, 0xf5, 0xeb, 0xe9, 0xef, 0xed, 0xe3, 0xe1, 0xe7, 0xe5
        ], dtype=self.dtype)

        self.mul3 = np.array([
            0x00, 0x03, 0x06, 0x05, 0x0c, 0x0f, 0x0a, 0x09, 0x18, 0x1b, 0x1e, 0x1d, 0x14, 0x17, 0x12, 0x11,
            0x30, 0x33, 0x36, 0x35, 0x3c, 0x3f, 0x3a, 0x39, 0x28, 0x2b, 0x2e, 0x2d, 0x24, 0x27, 0x22, 0x21,
            0x60, 0x63, 0x66, 0x65, 0x6c, 0x6f, 0x6a, 0x69, 0x78, 0x7b, 0x7e, 0x7d, 0x74, 0x77, 0x72, 0x71,
            0x50, 0x53, 0x56, 0x55, 0x5c, 0x5f, 0x5a, 0x59, 0x48, 0x4b, 0x4e, 0x4d, 0x44, 0x47, 0x42, 0x41,
            0xc0, 0xc3, 0xc6, 0xc5, 0xcc, 0xcf, 0xca, 0xc9, 0xd8, 0xdb, 0xde, 0xdd, 0xd4, 0xd7, 0xd2, 0xd1,
            0xf0, 0xf3, 0xf6, 0xf5, 0xfc, 0xff, 0xfa, 0xf9, 0xe8, 0xeb, 0xee, 0xed, 0xe4, 0xe7, 0xe2, 0xe1,
            0xa0, 0xa3, 0xa6, 0xa5, 0xac, 0xaf, 0xaa, 0xa9, 0xb8, 0xbb, 0xbe, 0xbd, 0xb4, 0xb7, 0xb2, 0xb1,
            0x90, 0x93, 0x96, 0x95, 0x9c, 0x9f, 0x9a, 0x99, 0x88, 0x8b, 0x8e, 0x8d, 0x84, 0x87, 0x82, 0x81,
            0x9b, 0x98, 0x9d, 0x9e, 0x97, 0x94, 0x91, 0x92, 0x83, 0x80, 0x85, 0x86, 0x8f, 0x8c, 0x89, 0x8a,
            0xab, 0xa8, 0xad, 0xae, 0xa7, 0xa4, 0xa1, 0xa2, 0xb3, 0xb0, 0xb5, 0xb6, 0xbf, 0xbc, 0xb9, 0xba,
            0xfb, 0xf8, 0xfd, 0xfe, 0xf7, 0xf4, 0xf1, 0xf2, 0xe3, 0xe0, 0xe5, 0xe6, 0xef, 0xec, 0xe9, 0xea,
            0xcb, 0xc8, 0xcd, 0xce, 0xc7, 0xc4, 0xc1, 0xc2, 0xd3, 0xd0, 0xd5, 0xd6, 0xdf, 0xdc, 0xd9, 0xda,
            0x5b, 0x58, 0x5d, 0x5e, 0x57, 0x54, 0x51, 0x52, 0x43, 0x40, 0x45, 0x46, 0x4f, 0x4c, 0x49, 0x4a,
            0x6b, 0x68, 0x6d, 0x6e, 0x67, 0x64, 0x61, 0x62, 0x73, 0x70, 0x75, 0x76, 0x7f, 0x7c, 0x79, 0x7a,
            0x3b, 0x38, 0x3d, 0x3e, 0x37, 0x34, 0x31, 0x32, 0x23, 0x20, 0x25, 0x26, 0x2f, 0x2c, 0x29, 0x2a,
            0x0b, 0x08, 0x0d, 0x0e, 0x07, 0x04, 0x01, 0x02, 0x13, 0x10, 0x15, 0x16, 0x1f, 0x1c, 0x19, 0x1a
        ], dtype=self.dtype)

        # self.counter = self.dtype(14111985)

        self.entropy_kernel = self.compile_entropy_kernel()

    @staticmethod
    def __read_file(path, mode="r"):
        with open(path, mode) as f:
            return f.read()

    def compile_entropy_kernel(self):
        kernel_code = """
        __global__ void calculate_entropy(unsigned char *data, int *frequency, int size) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < size) {
                atomicAdd(&frequency[data[idx]], 1);
            }
        }

        __global__ void compute_entropy(int *frequency, float *entropy, int total_count) {
            int idx = threadIdx.x;
            if (idx < 256) {
                if (frequency[idx] > 0) {
                    float prob = (float)frequency[idx] / total_count;
                    atomicAdd(entropy, -prob * log2(prob));
                }
            }
        }
        """
        mod = SourceModule(kernel_code)
        return mod.get_function("calculate_entropy"), mod.get_function("compute_entropy")

    def calculate_entropy(self, encrypted_data):
        # Transfer data to the GPU
        data_gpu = cuda.mem_alloc(encrypted_data.nbytes)
        cuda.memcpy_htod(data_gpu, encrypted_data)

        # Initialize frequency on the GPU
        frequency = np.zeros(256, dtype=np.int32)
        frequency_gpu = cuda.mem_alloc(frequency.nbytes)
        cuda.memcpy_htod(frequency_gpu, frequency)

        # Calculate entropy
        size = encrypted_data.size
        block_size = 256  # You can adjust this size as needed
        grid_size = (size + block_size - 1) // block_size

        # Count frequencies
        calculate_entropy_kernel, compute_entropy_kernel = self.compile_entropy_kernel()
        calculate_entropy_kernel(data_gpu, frequency_gpu, np.int32(size), block=(block_size, 1, 1),
                                 grid=(grid_size, 1, 1))

        # Copy frequencies back to the CPU
        cuda.memcpy_dtoh(frequency, frequency_gpu)

        # Calculate the total count
        total_count = np.sum(frequency)
        entropy_value = np.zeros(1, dtype=np.float32)
        entropy_gpu = cuda.mem_alloc(entropy_value.nbytes)
        cuda.memcpy_htod(entropy_gpu, entropy_value)

        # Compute entropy using the kernel
        compute_entropy_kernel(frequency_gpu, entropy_gpu, np.int32(total_count), block=(256, 1, 1), grid=(1, 1, 1))

        # Copy the entropy value back to the CPU
        cuda.memcpy_dtoh(entropy_value, entropy_gpu)

        return entropy_value[0]

    def get_source_module_encrypt(self):
        """
        
        """
        private_sharedlut = """
        #define AES_PRIVATESTATE_SHAREDLUT
        #define LUT_IN_SHARED
        """

        kernelwrapper = self.__read_file("kernels/general.cuh")
        kernelwrapper += self.__read_file("kernels/SubBytes.cuh")
        kernelwrapper += self.__read_file("kernels/ShiftRows.cuh")
        kernelwrapper += self.__read_file("kernels/MixColumns.cuh")
        kernelwrapper += self.__read_file("kernels/AddRoundKey.cuh")
        kernelwrapper += self.__read_file("kernels/Round.cuh")
        kernelwrapper += self.__read_file("kernels/KeyExpansion.cuh")
        kernelwrapper += self.__read_file("kernels/FinalRound.cuh")
        kernelwrapper += self.__read_file("kernels/AES.cuh")

        self.module_encrypt = SourceModule(private_sharedlut + kernelwrapper)

    def get_source_module_decrypt(self):
        """

        """
        sharedLut = """
        #define LUT_IN_SHARED
        """
        kernelwrapper = self.__read_file("kernels/general.cuh")
        kernelwrapper += self.__read_file("kernels/InvSubbytes.cuh")
        kernelwrapper += self.__read_file("kernels/InvShiftRows.cuh")
        kernelwrapper += self.__read_file("kernels/MixColumns.cuh")
        kernelwrapper += self.__read_file("kernels/InvMixColumns.cuh")
        kernelwrapper += self.__read_file("kernels/AddRoundKey.cuh")
        kernelwrapper += self.__read_file("kernels/InvRound.cuh")
        kernelwrapper += self.__read_file("kernels/KeyExpansion.cuh")
        kernelwrapper += self.__read_file("kernels/InvFinalRound.cuh")

        kernelwrapper += self.__read_file("kernels/SubBytes.cuh")
        kernelwrapper += self.__read_file("kernels/ShiftRows.cuh")
        kernelwrapper += self.__read_file("kernels/Round.cuh")
        kernelwrapper += self.__read_file("kernels/FinalRound.cuh")

        kernelwrapper += self.__read_file("kernels/InvAES.cuh")

        self.module_decrpyt = SourceModule(sharedLut + kernelwrapper)

    def encrypt_gpu(self, state, cipherkey, block_size=None):
        state = np.frombuffer(state, dtype=self.dtype)  # Convert bytes to numpy array

        # Pad the message so its length is a multiple of 16 bytes
        padding_len = 16 - (state.size % 16)
        if padding_len != 16:
            logger.warning("Padding length: %d", padding_len)
            padding = np.zeros(padding_len, dtype=self.dtype)
            state = np.concatenate((state, padding))

        logger.warning("Partial state: %s", state[:16])

        # Device memory allocation for input and output arrays
        io_state_gpu = cuda.mem_alloc_like(state)
        i_cipherkey_gpu = cuda.mem_alloc_like(cipherkey)
        i_rcon_gpu = cuda.mem_alloc_like(self.rcon)
        i_sbox_gpu = cuda.mem_alloc_like(self.sbox)
        i_mul2_gpu = cuda.mem_alloc_like(self.mul2)
        i_mul3_gpu = cuda.mem_alloc_like(self.mul3)

        # Copy data from host to device
        cuda.memcpy_htod(io_state_gpu, state)
        cuda.memcpy_htod(i_cipherkey_gpu, cipherkey)
        cuda.memcpy_htod(i_rcon_gpu, self.rcon)
        cuda.memcpy_htod(i_sbox_gpu, self.sbox)
        cuda.memcpy_htod(i_mul2_gpu, self.mul2)
        cuda.memcpy_htod(i_mul3_gpu, self.mul3)

        # Calculate block size and grid size
        if block_size is None:
            block_size = (state.size - 1) // 16 + 1
            grid_size = 1
            if (block_size > 1024):
                block_size = 1024
                grid_size = (state.size - 1) // (1024 * 16) + 1
        else:
            grid_size = (state.size - 1) // (block_size * 16) + 1

        blockDim = (block_size, 1, 1)
        gridDim = (grid_size, 1, 1)

        # call kernel
        prg = self.module_encrypt.get_function("AES_private_sharedlut")
        prg(io_state_gpu, i_cipherkey_gpu, np.uint32(state.size), i_rcon_gpu, i_sbox_gpu, i_mul2_gpu, i_mul3_gpu,
            block=blockDim, grid=gridDim)

        # copy results from device to host
        res = np.empty_like(state)
        cuda.memcpy_dtoh(res, io_state_gpu)

        del io_state_gpu, i_cipherkey_gpu, i_rcon_gpu, i_sbox_gpu, i_mul2_gpu, i_mul3_gpu

        logger.warning("Partial res: %s", res[:16])
        # Return the result
        return res

    def decrypt_gpu(self, state, cipherkey, block_size=None):
        state = np.frombuffer(state, dtype=self.dtype)  # Convert bytes to numpy array

        # Pad the message so its length is a multiple of 16 bytes
        padding_len = 16 - (state.size % 16)
        if padding_len != 16:
            logger.warning("Padding length: %d", padding_len)
            padding = np.zeros(padding_len, dtype=self.dtype)
            state = np.concatenate((state, padding))

        logger.warning("Partial state: %s", state[:16])

        # device memory allocation
        io_state_gpu = cuda.mem_alloc_like(state)
        i_cipherkey_gpu = cuda.mem_alloc_like(cipherkey)
        i_rcon_gpu = cuda.mem_alloc_like(self.rcon)
        i_sbox_gpu = cuda.mem_alloc_like(self.sbox)
        i_invsbox_gpu = cuda.mem_alloc_like(self.invSbox)
        i_mul2_gpu = cuda.mem_alloc_like(self.mul2)
        i_mul3_gpu = cuda.mem_alloc_like(self.mul3)

        cuda.memcpy_htod(io_state_gpu, state)
        cuda.memcpy_htod(i_cipherkey_gpu, cipherkey)
        cuda.memcpy_htod(i_rcon_gpu, self.rcon)
        cuda.memcpy_htod(i_sbox_gpu, self.sbox)
        cuda.memcpy_htod(i_invsbox_gpu, self.invSbox)
        cuda.memcpy_htod(i_mul2_gpu, self.mul2)
        cuda.memcpy_htod(i_mul3_gpu, self.mul3)

        # Calculate block size and grid size
        if block_size is None:
            block_size = (state.size - 1) // 16 + 1
            grid_size = 1
            if (block_size > 1024):
                block_size = 1024
                grid_size = (state.size - 1) // (1024 * 16) + 1
        else:
            grid_size = (state.size - 1) // (block_size * 16) + 1

        blockDim = (block_size, 1, 1)
        gridDim = (grid_size, 1, 1)

        # call kernel
        prg = self.module_decrpyt.get_function("inv_AES")
        prg(io_state_gpu, i_cipherkey_gpu, np.uint32(state.size), i_rcon_gpu, i_sbox_gpu, i_invsbox_gpu,
            i_mul2_gpu, i_mul3_gpu, block=blockDim, grid=gridDim)

        # Copy result from device to the host
        res = np.empty_like(state)
        cuda.memcpy_dtoh(res, io_state_gpu)

        # Remove padding
        res = res[:-padding_len] if padding_len != 16 else res

        del io_state_gpu, i_cipherkey_gpu, i_rcon_gpu, i_sbox_gpu, i_invsbox_gpu, i_mul2_gpu, i_mul3_gpu

        logger.warning("Partial res: %s", res[:16])
        return res

    def encrypt_ctr_gpu(self, state, cipherkey, counterinit, block_size=None):
        state = np.frombuffer(state, dtype=self.dtype)  # Convert bytes to numpy array

        # Pad the message so its length is a multiple of 16 bytes
        padding_len = 16 - (state.size % 16)
        if padding_len != 16:
            logger.warning("Padding length: %d", padding_len)
            padding = np.zeros(padding_len, dtype=self.dtype)
            state = np.concatenate((state, padding))

        logger.warning("Partial state: %s", state[:16])

        # Device memory allocation for input and output arrays
        io_state_gpu = cuda.mem_alloc_like(state)
        i_cipherkey_gpu = cuda.mem_alloc_like(cipherkey)
        i_rcon_gpu = cuda.mem_alloc_like(self.rcon)
        i_sbox_gpu = cuda.mem_alloc_like(self.sbox)
        i_mul2_gpu = cuda.mem_alloc_like(self.mul2)
        i_mul3_gpu = cuda.mem_alloc_like(self.mul3)

        # Copy data from host to device
        cuda.memcpy_htod(io_state_gpu, state)
        cuda.memcpy_htod(i_cipherkey_gpu, cipherkey)
        cuda.memcpy_htod(i_rcon_gpu, self.rcon)
        cuda.memcpy_htod(i_sbox_gpu, self.sbox)
        cuda.memcpy_htod(i_mul2_gpu, self.mul2)
        cuda.memcpy_htod(i_mul3_gpu, self.mul3)

        # Calculate block size and grid size
        if block_size is None:
            block_size = (state.size - 1) // 16 + 1
            grid_size = 1
            if (block_size > 1024):
                block_size = 1024
                grid_size = (state.size - 1) // (1024 * 16) + 1
        else:
            grid_size = (state.size - 1) // (block_size * 16) + 1

        blockDim = (block_size, 1, 1)
        gridDim = (grid_size, 1, 1)

        # call kernel
        prg = self.module_encrypt.get_function("AES_CTR")
        prg(io_state_gpu, i_cipherkey_gpu, np.uint32(state.size), i_rcon_gpu, i_sbox_gpu, i_mul2_gpu, i_mul3_gpu,
            np.uint32(counterinit),
            block=blockDim, grid=gridDim)

        # copy results from device to host
        res = np.empty_like(state)
        cuda.memcpy_dtoh(res, io_state_gpu)

        del io_state_gpu, i_cipherkey_gpu, i_rcon_gpu, i_sbox_gpu, i_mul2_gpu, i_mul3_gpu

        logger.warning("Partial res: %s", res[:16])
        # Return the result
        return res


class AESStats():
    def __init__(self, fits_file, key):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.info("************ AES %s ************", fits_file)
        self.key = key
        self.fits_file = fits_file
        self.fits_data = None
        self.fits_header = None
        self.counter = 14111985

    def set_log_level(self, level):
        self.logger.setLevel(level)

    def get_data(self):
        if self.fits_data is None:
            start_time = time.time()
            self.fits_data = fits.getdata(self.fits_file)
            logger.info("Data loaded in %.2f seconds", time.time() - start_time)
            logger.info("Data shape: %s", self.fits_data.shape)
        return self.fits_data

    def get_header(self):
        if self.fits_header is None:
            self.fits_header = fits.getheader(self.fits_file).tostring()
        return self.fits_header

    def data_to_bytes(self):
        return self.get_data().tobytes()

    def header_to_bytes(self):
        return self.get_header().encode('utf-8')

    def frombuffer_data(self, data_bytes):
        return np.frombuffer(data_bytes, dtype=self.get_data().dtype).reshape(self.get_data().shape)

    def frombuffer_header(self, header_bytes):
        return np.frombuffer(header_bytes, dtype=np.byte)

    def compute_aes_ebs_data(self):
        self.logger.info("*** AES EBC ***")
        aes = AES()
        start_time = time.time()
        data_bytes = self.data_to_bytes()
        key_array = np.frombuffer(self.key, dtype=np.byte)
        start_time_encryption = time.time()
        encrypt_bytes = aes.encrypt_gpu(data_bytes, key_array)
        encrypted_data = self.frombuffer_data(encrypt_bytes)
        logger.info("Encryption complete in %.2f seconds", time.time() - start_time_encryption)

        start_time_decryption = time.time()
        decrypt_bytes = aes.decrypt_gpu(encrypt_bytes, key_array)
        decrypted_data = self.frombuffer_data(decrypt_bytes)
        logger.info("Decryption complete in %.2f seconds", time.time() - start_time_decryption)

        logger.info("Total time: %.2f seconds", time.time() - start_time)
        return encrypted_data, decrypted_data

    def compute_aes_ctr_data(self):
        self.logger.info("*** AES CTR ***")
        aes = AES()
        start_time = time.time()
        data_bytes = self.data_to_bytes()
        key_array = np.frombuffer(self.key, dtype=np.byte)
        start_time_encryption = time.time()
        encrypt_bytes = aes.encrypt_ctr_gpu(data_bytes, key_array, self.counter)
        encrypted_data = self.frombuffer_data(encrypt_bytes)
        logger.info("Encryption complete in %.2f seconds", time.time() - start_time_encryption)

        start_time_decryption = time.time()
        decrypt_bytes = aes.encrypt_ctr_gpu(encrypt_bytes, key_array, self.counter)
        decrypted_data = self.frombuffer_data(decrypt_bytes)
        logger.info("Decryption complete in %.2f seconds", time.time() - start_time_decryption)

        logger.info("Total time: %.2f seconds", time.time() - start_time)
        return encrypted_data, decrypted_data

    def compute_aes_ebs_header(self):
        self.logger.info("*** AES EBC ***")
        aes = AES()
        start_time = time.time()
        header_bytes = self.header_to_bytes()
        key_array = np.frombuffer(self.key, dtype=np.byte)
        start_time_encryption = time.time()
        encrypt_bytes = aes.encrypt_gpu(header_bytes, key_array)
        encrypt_bytes_hex = bytes(encrypt_bytes).hex()
        logger.info("Encryption complete in %.2f seconds", time.time() - start_time_encryption)

        start_time_decryption = time.time()
        decrypt_bytes = aes.decrypt_gpu(encrypt_bytes, key_array)
        decrypt_bytes = "".join([chr(item) for item in decrypt_bytes])
        decrypt_bytes = decrypt_bytes[:len(self.get_header())]
        logger.info("Decryption complete in %.2f seconds", time.time() - start_time_decryption)

        logger.info("Total time: %.2f seconds", time.time() - start_time)
        return encrypt_bytes_hex, decrypt_bytes

    def compute_aes_ctr_header(self):
        self.logger.info("*** AES CTR ***")
        aes = AES()
        start_time = time.time()
        header_bytes = self.header_to_bytes()
        key_array = np.frombuffer(self.key, dtype=np.byte)
        start_time_encryption = time.time()
        encrypt_bytes = aes.encrypt_ctr_gpu(header_bytes, key_array, self.counter)
        encrypt_bytes_hex = bytes(encrypt_bytes).hex()
        logger.info("Encryption complete in %.2f seconds", time.time() - start_time_encryption)

        start_time_decryption = time.time()
        decrypt_bytes = aes.encrypt_ctr_gpu(encrypt_bytes, key_array, self.counter)
        decrypt_bytes = "".join([chr(item) for item in decrypt_bytes])
        decrypt_bytes = decrypt_bytes[:len(self.get_header())]
        logger.info("Decryption complete in %.2f seconds", time.time() - start_time_decryption)

        logger.info("Total time: %.2f seconds", time.time() - start_time)
        return encrypt_bytes_hex, decrypt_bytes

    def analyze_entropy(self, data):
        aes = AES()
        entropy = aes.calculate_entropy(data)
        logger.info("Entropy: %s", entropy)

    def check_results_data(self, decrypted_data, encrypted_data):
        if np.array_equal(decrypted_data, encrypted_data):
            logger.error("#######    Decrypted data is equal to the encrypted data")
        else:
            if np.allclose(decrypted_data, encrypted_data):
                logger.error("#######    Decrypted data is close to the encrypted data")
            else:
                if self.get_data().all() != decrypted_data.all():
                    logger.error("#######    Decrypted data is different to the original data, Shapes: %s, %s",
                                 decrypted_data.shape, self.get_data().shape)
                else:
                    pass
                    # logger.info("Decrypted data is different from the original data")

    def analyze_results_data(self, encrypted_data, decrypted_data):
        self.check_results_data(decrypted_data, encrypted_data)
        self.analyze_entropy(encrypted_data)
        self.analyze_entropy(decrypted_data)

    def analyze_aes(self):
        logger.info(" AES Data Analysis")
        encrypted_data_ebs, decrypted_data_ebs = self.compute_aes_ebs_data()
        self.analyze_results_data(encrypted_data_ebs, decrypted_data_ebs)
        encrypted_data_ctr, decrypted_data_ctr = self.compute_aes_ctr_data()
        self.analyze_results_data(encrypted_data_ctr, decrypted_data_ctr)

        logger.info(" AES Header Analysis")
        encrypted_header_ebs, decrypted_header_ebs = self.compute_aes_ebs_header()
        self.analyze_results_header(encrypted_header_ebs, decrypted_header_ebs)
        encrypted_header_ctr, decrypted_header_ctr = self.compute_aes_ctr_header()
        self.analyze_results_header(encrypted_header_ctr, decrypted_header_ctr)

        logger.info("************ END %s ************", self.fits_file)

    def analyze_results_header(self, encrypted_header_ebs, decrypted_header_ebs):
        self.check_results_header(decrypted_header_ebs, encrypted_header_ebs)

    def check_results_header(self, decrypted_header_ebs, encrypted_header_ebs):

        # Comparación de arrays usando numpy.array_equal
        if np.array_equal(decrypted_header_ebs, encrypted_header_ebs):
            logger.error("#######    Decrypted header is equal to the encrypted header")
        else:
            if not np.array_equal(self.get_header(), decrypted_header_ebs):
                logger.error("#######    Decrypted header is different to the original header")

                # Comparar las longitudes
                if len(decrypted_header_ebs) != len(self.get_header()):
                    logger.error(
                        f"Longitud diferente: decrypted_header_ebs({len(decrypted_header_ebs)}) != original_header({len(self.get_header())})")
                else:
                    # Identificar y reportar las diferencias en contenido
                    diferencias = np.where(decrypted_header_ebs != self.get_header())
                    logger.error(f"Diferencias en índices: {diferencias[0]}")
                    for idx in diferencias[0]:
                        logger.error(
                            f"Diferencia en índice {idx}: decrypted_header_ebs={decrypted_header_ebs[idx]}, original_header={self.get_header()[idx]}")
            else:
                pass
                # logger.info("Decrypted header is different from the original header")


#
# def main_aes_ctr(fits_file):
#     logger.info("************ AES CTR ************")
#     logger.info("Loading FITS file: %s", fits_file)
#
#     # Medir tiempo de carga de datos
#     start_time = time.time()
#     try:
#         image_data = fits.getdata(fits_file)
#         load_data_time = time.time() - start_time
#         logger.info("Data loaded in %.2f seconds", load_data_time)
#     except Exception as e:
#         logger.error("Error loading FITS file: %s", e)
#         raise
#
#     # Medir tiempo de conversión a bytes
#     start_time = time.time()
#     try:
#         image_bytes = image_data.tobytes()
#         tobytes_time = time.time() - start_time
#         logger.info("Conversion to bytes completed in %.2f seconds", tobytes_time)
#     except Exception as e:
#         logger.error("Error converting image data to bytes: %s", e)
#         raise
#
#     # Generar clave aleatoria
#     byte_key = os.urandom(128)
#
#     # # Inicializar el gestor de claves
#     # master_key = b'pruebaKeyMaster'  # Asegúrate de que esto sea una clave segura en producción
#     # key_manager = FileEncryptor(master_key)
#     #
#     # # Generar clave de usuario
#     # user_id = b'user123'  # Este podría ser un identificador único para cada usuario
#     # byte_array_key_user = key_manager.get_user_key(user_id)
#     # # byte_array_key_master = key_manager.get_master_key()
#
#     byte_array_key = np.frombuffer(byte_key, dtype=np.byte)
#
#     # Crear instancia de AES
#     computer = AES()
#
#     # Encriptar los datos de la imagen
#     logger.info("Encrypting the input...")
#     start_time = time.time()
#     encrypted_bytes = computer.encrypt_ctr_gpu(image_bytes, byte_array_key)
#     encryption_time = time.time() - start_time
#     logger.info("Encryption complete in %.2f seconds", encryption_time)
#
#     # Convertir los bytes encriptados a un array de numpy
#     encrypted_data = np.frombuffer(encrypted_bytes, dtype=image_data.dtype).reshape(image_data.shape)
#
#     # Desencriptar los datos de la imagen
#     logger.info("Decrypting the input...")
#     start_time = time.time()
#     decrypted_bytes = computer.encrypt_ctr_gpu(encrypted_bytes, byte_array_key)
#     decryption_time = time.time() - start_time
#     logger.info("Decryption complete in %.2f seconds", decryption_time)
#
#     # # Desencriptar los datos de la imagen
#     # logger.info("Decrypting the input...")
#     # start_time = time.time()
#     # decrypted_bytes_master = computer.encrypt_ctr_gpu(encrypted_bytes, byte_array_key_master)
#     # decryption_time = time.time() - start_time
#     # logger.info("Decryption complete in %.2f seconds", decryption_time)
#
#     # Convertir los bytes desencriptados a un array de numpy
#     decrypted_data = np.frombuffer(decrypted_bytes, dtype=image_data.dtype).reshape(image_data.shape)
#     # decrypted_data_master = np.frombuffer(decrypted_bytes_master, dtype=image_data.dtype).reshape(image_data.shape)
#
#     check_results(decrypted_data, encrypted_data, image_data)
#     # check_results(decrypted_data_master, encrypted_data, image_data)
#
#     result_entropy = computer.calculate_entropy(encrypted_data)
#     logger.info("Entropy: %s", result_entropy)
#
#
# def main_aes(fits_file):
#     logger.info("************ AES ************")
#     logger.info("Loading FITS file: %s", fits_file)
#
#     # Medir tiempo de carga de datos
#     start_time = time.time()
#     try:
#         image_data = fits.getdata(fits_file)
#         load_data_time = time.time() - start_time
#         logger.info("Data loaded in %.2f seconds", load_data_time)
#     except Exception as e:
#         logger.error("Error loading FITS file: %s", e)
#         raise
#
#     # Medir tiempo de conversión a bytes
#     start_time = time.time()
#     try:
#         image_bytes = image_data.tobytes()
#         tobytes_time = time.time() - start_time
#         logger.info("Conversion to bytes completed in %.2f seconds", tobytes_time)
#     except Exception as e:
#         logger.error("Error converting image data to bytes: %s", e)
#         raise
#
#     # Generar clave aleatoria
#     byte_key = os.urandom(128)
#     byte_array_key = np.frombuffer(byte_key, dtype=np.byte)
#
#     # Crear instancia de AES
#     computer = AES()
#
#     # Encriptar los datos de la imagen
#     logger.info("Encrypting the input...")
#     start_time = time.time()
#     encrypted_bytes = computer.encrypt_gpu(image_bytes, byte_array_key)
#     encryption_time = time.time() - start_time
#     logger.info("Encryption complete in %.2f seconds", encryption_time)
#
#     # Convertir los bytes encriptados a un array de numpy
#     encrypted_data = np.frombuffer(encrypted_bytes, dtype=image_data.dtype).reshape(image_data.shape)
#
#     # Desencriptar los datos de la imagen
#     logger.info("Decrypting the input...")
#     start_time = time.time()
#     decrypted_bytes = computer.decrypt_gpu(encrypted_bytes, byte_array_key)
#     decryption_time = time.time() - start_time
#     logger.info("Decryption complete in %.2f seconds", decryption_time)
#
#     # Convertir los bytes desencriptados a un array de numpy
#     decrypted_data = np.frombuffer(decrypted_bytes, dtype=image_data.dtype).reshape(image_data.shape)
#
#     check_results(decrypted_data, encrypted_data, image_data)
#
#     result_entropy = computer.calculate_entropy(encrypted_data)
#     logger.info("Entropy: %s", result_entropy)
#     # plot_interactive_image(image_data, encrypted_data, decrypted_data)


def check_results(decrypted_data, encrypted_data, image_data):
    import numpy as np
    # plot_interactive_image(image_data, encrypted_data, decrypted_data, crop_size=50)
    # Verificar si la imagen original es igual a la desencriptada
    data_is_equal_ori_de = np.array_equal(image_data, decrypted_data)
    data_is_equal_ori_en = np.array_equal(image_data, encrypted_data)
    data_is_equal_en_de = np.array_equal(encrypted_data, decrypted_data)

    if not data_is_equal_ori_de:
        logger.info("********** Image data is equal to decrypted data: %s", data_is_equal_ori_de)

    if data_is_equal_ori_en:
        logger.info("********** Image data is equal to encrypted data: %s", data_is_equal_ori_en)

    if data_is_equal_en_de:
        logger.info("********** Encrypted data is equal to decrypted data: %s", data_is_equal_en_de)

    if data_is_equal_ori_de and not data_is_equal_en_de and not data_is_equal_ori_en:
        logger.info("Encrypted data is not equal to decrypted data")

    plot_interactive_image(image_data, encrypted_data, decrypted_data)


if __name__ == "__main__":
    import logging
    import time
    from astropy.io import fits

    # Configuración del logger
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Autoinit device: %s", autoinit.device)

    # # Archivo FITS
    # fits_file = './Images/TTT1_iKon936-1_2024-05-12-23-55-18-657530_Ton599.fits'
    # # fits_file = './Images/TTT2_QHY411-2_2024-05-13-03-02-54-564461_Chariklo.fits'
    #
    # main_aes(fits_file)
    # main_aes_ctr(fits_file)
    #

    import time
    import traceback
    from astropy.io import fits
    import os

    # Root image directory
    directory_path = os.path.join(os.path.dirname(__file__), 'Images')
    directory_path = os.path.join('/home/slemes/PycharmProjects/GPUPhotFinal/tests/data')

    # Get all subdirectories and find if exists a FITS file
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.fits'):
                if "TTT1" in file:
                    image_path = os.path.join(root, file)
                    try:
                        logger.info(f"Processing {image_path}")
                        start_time = time.time()

                        testing_key = b'pruebaKeyMaster'

                        aes_stats = AESStats(image_path, testing_key)
                        # aes_stats.set_log_level(logging.WARNING)
                        aes_stats.analyze_aes()
                        # ctr_encrypt, ctr_decrypt = aes_stats.compute_aes_ctr(testing_key)
                        # ebs_encrypt, ebs_decrypt = aes_stats.compute_aes_ebs(testing_key)
                        #
                        # if ctr_decrypt.all() == ebs_decrypt.all() and ebs_decrypt.all() == aes_stats.get_data().all():
                        #     logger.info("Decrypted Works Fine for file: %s", file)
                        # elif ctr_decrypt.all() != ebs_decrypt.all():
                        #     if ctr_decrypt.all() != aes_stats.get_data().all():
                        #         logger.error("Decrypted data is not equal to original data for file: %s", file)
                        #     if ebs_decrypt.all() != aes_stats.get_data().all():
                        #         logger.error("Decrypted data is not equal to original data for file: %s", file)
                        # else:
                        #     logger.error("Decrypted data is not equal to original data for file: %s", file)

                        end_time = time.time()
                        logger.info(f"Elapsed time for {file}: {end_time - start_time:.2f} seconds")
                    except Exception as e:
                        logger.error(f"Error processing {image_path}: {e}")
                        logger.error("Traceback:")
                        traceback.print_exc()
                break

# clave secreta -> Datos de la imagen
# esquema umbral 2:n (n variable?) (sistemas de ecuaciones) -> Header imagen
"""
Tiempos de generar en distintos tipos
Medir entropía de la imagen


AES
Generación claves aes
esquema umbral para proteger claves


Generación de claves AES
Esquema kem 
Crystals KEM postcuantica  kyber
"""
