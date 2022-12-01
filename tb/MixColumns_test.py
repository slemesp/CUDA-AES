import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

class MixColumnsTest:
    def __init__(self):
        self.getSourceModule()

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
            ], dtype=np.byte)

    def getSourceModule(self):
        with open("../kernels/MixColumns.cuh",  "r") as file:
            kernelWrapper = file.read()

        enable_test = r"""
        #define TEST_MIXCOLUMNS
        """

        self.module = SourceModule(enable_test + kernelWrapper)

    def mixcolumns_gpu(self, message, length):
        # Event objects to mark the start and end points
        start = cuda.Event()
        end = cuda.Event()

        # Start recording execution time
        start.record()

        # Device memory allocation for input and output arrays
        io_message_gpu = cuda.mem_alloc_like(message)
        i_mem3_gpu = self.module.get_global('mul3')[0]

        # Copy data from host to device
        cuda.memcpy_htod(io_message_gpu, message)
        cuda.memcpy_htod(i_mem3_gpu, self.mul3)

        # Call the kernel function from the compiled module
        prg = self.module.get_function("mixColumnsTest")

        # Calculate block size and grid size
        block_size = length
        grid_size = 1
        if (block_size > 1024):
            block_size = 1024
            grid_size = (length - 1) / 1024 + 1;

        blockDim = (block_size, 1, 1)
        gridDim = (grid_size, 1, 1)

        # Call the kernel loaded to the device
        prg(io_message_gpu, np.uint32(length), block=blockDim, grid=gridDim)

        # Copy result from device to the host
        res = np.empty_like(message)
        cuda.memcpy_dtoh(res, io_message_gpu)

        # Record execution time (including memory transfers)
        end.record()
        end.synchronize()

        # return a tuple of output of sine computation and time taken to execute the operation (in ms).
        return res, start.time_till(end) * 1e-3

def test_xtime():
    messages = np.array([0x57, 0xae, 0x47, 0x8e], dtype=np.byte)

    graphicsComputer = MixColumnsTest()

    # Device memory allocation for input and output arrays
    io_message_gpu = cuda.mem_alloc(messages.size * messages.dtype.itemsize)

    # Copy data from host to device
    cuda.memcpy_htod(io_message_gpu, messages)

    # Call the kernel function from the compiled module
    prg = graphicsComputer.module.get_function("xtime_test")

    # Calculate block size and grid size
    block_size = 1
    grid_size = messages.size
    blockDim = (block_size, 1, 1)
    gridDim = (grid_size, 1, 1)

    # Call the kernel loaded to the device
    prg(io_message_gpu, block=blockDim, grid=gridDim)

    # Copy result from device to the host
    res = np.empty_like(messages)
    cuda.memcpy_dtoh(res, io_message_gpu)
    
    print(res)
    print(messages)

    assert np.array_equal(res, np.array([0xae, 0x47, 0x8e, 0x07], dtype=np.byte))

def test1_mixColumns():
    # input array
    hex_in = "6353e08c0960e104cd70b751bacad0e7" 
    byte_in = bytes.fromhex(hex_in)
    byte_array_in = np.frombuffer(byte_in, dtype=np.byte)

    # reference input
    hex_ref = "5f72641557f5bc92f7be3b291db9f91a"
    byte_ref = bytes.fromhex(hex_ref)
    byte_array_ref = np.frombuffer(byte_ref, dtype=np.byte)
    
    graphicsComputer = MixColumnsTest()
    result_gpu, _ = graphicsComputer.mixcolumns_gpu(byte_array_in, byte_array_in.size)

    print(byte_array_ref)
    print(byte_array_in)
    print(result_gpu)

    assert np.array_equal(result_gpu, byte_array_ref)

def test2_mixColumns():
    # input array
    hex_in = "3bd92268fc74fb735767cbe0c0590e2d" 
    byte_in = bytes.fromhex(hex_in)
    byte_array_in = np.frombuffer(byte_in, dtype=np.byte)

    # reference input
    hex_ref = "4c9c1e66f771f0762c3f868e534df256"
    byte_ref = bytes.fromhex(hex_ref)
    byte_array_ref = np.frombuffer(byte_ref, dtype=np.byte)

    graphicsComputer = MixColumnsTest()
    result_gpu, _ = graphicsComputer.mixcolumns_gpu(byte_array_in, byte_array_in.size)

    print(byte_array_ref)
    # print(byte_array_in)
    print(result_gpu)

    assert np.array_equal(result_gpu, byte_array_ref)