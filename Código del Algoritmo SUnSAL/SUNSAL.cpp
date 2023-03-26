#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

#include "SUNSAL.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>

static const int DATA_SIZE = 131072;
static const int K = 5;
static const int N = 3;

static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";

int main(int argc, char* argv[]) {
    // TARGET_DEVICE macro needs to be passed from gcc command line
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <xclbin>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string xclbinFilename = argv[1];

    // Compute the size of array in bytes
    size_t size_in_bytes = DATA_SIZE * sizeof(float);

    // Creates a vector of DATA_SIZE elements with an initial value of 10 and 32
    // using customized allocator for getting buffer alignment to 4k boundary

    std::vector<cl::Device> devices;
    cl_int err;
    cl::Context context;
    cl::CommandQueue q;
    cl::Kernel krnl_SUNSAL;
    cl::Program program;
    std::vector<cl::Platform> platforms;
    bool found_device = false;

    // traversing all Platforms To find Xilinx Platform and targeted
    // Device in Xilinx Platform
    cl::Platform::get(&platforms);
    for (size_t i = 0; (i < platforms.size()) & (found_device == false); i++) {
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if (platformName == "Xilinx") {
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
            if (devices.size()) {
                found_device = true;
                break;
            }
        }
    }
    if (found_device == false) {
        std::cout << "Error: Unable to find Target Device " << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "INFO: Reading " << xclbinFilename << std::endl;
    FILE* fp;
    if ((fp = fopen(xclbinFilename.c_str(), "r")) == nullptr) {
        printf("ERROR: %s xclbin not available please build\n", xclbinFilename.c_str());
        exit(EXIT_FAILURE);
    }
    // Load xclbin
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    char* buf = new char[nb];
    bin_file.read(buf, nb);

    // Creating Program from Binary File
    cl::Program::Binaries bins;
    bins.push_back({buf, nb});
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_SUNSAL = cl::Kernel(program, "krnl_SUNSAL", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    // These commands will allocate memory on the Device. The cl::Buffer objects can
    // be used to reference the memory locations on the device.
    OCL_CHECK(err, cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, size_in_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_y(context, CL_MEM_READ_ONLY, size_in_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_x(context, CL_MEM_WRITE_ONLY, size_in_bytes, NULL, &err));

    // set the kernel Arguments
    int narg = 0;
    OCL_CHECK(err, err = krnl_SUNSAL.setArg(narg++, buffer_A));
    OCL_CHECK(err, err = krnl_SUNSAL.setArg(narg++, buffer_y));
    OCL_CHECK(err, err = krnl_SUNSAL.setArg(narg++, buffer_x));

    // We then need to map our OpenCL buffers to get the pointers
    float* ptr_A;
    float* ptr_y;
    float* ptr_x;
    OCL_CHECK(err, ptr_A = (float*)q.enqueueMapBuffer(buffer_A, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes, NULL, NULL, &err));
    OCL_CHECK(err, ptr_y = (float*)q.enqueueMapBuffer(buffer_y, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes, NULL, NULL, &err));
    OCL_CHECK(err, ptr_x = (float*)q.enqueueMapBuffer(buffer_x, CL_TRUE, CL_MAP_READ, 0, size_in_bytes, NULL, NULL, &err));

    // Matrix A (K * N)
    ptr_A[0] = 2; ptr_A[1] = 1; ptr_A[2] = 4; ptr_A[3] = 0; ptr_A[4] = 2; ptr_A[5] = 0;
    ptr_A[6] = 1; ptr_A[7] = 1; ptr_A[8] = 0; ptr_A[9] = 0; ptr_A[10] = 3; ptr_A[11] = 1;
    ptr_A[12] = 1; ptr_A[13] = 5; ptr_A[14] = 0;
    // Vector y (K)
    ptr_y[0] = 2; ptr_y[1] = 3; ptr_y[2] = 1; ptr_y[3] = 3; ptr_y[4] = 1;
    std::cout << "Memory Allocated Correctly" << "\n";

    /*
    for(int i = 0; i < K*N; i++) {
    	ptr_A[i] = 1;
    }

    for(int i = 0; i < K; i++) {
    	ptr_y[i] = 1;
    }
    */

    // Data will be migrated to kernel space
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_A, buffer_y}, 0 /* 0 means from host*/));

    // Launch the Kernel
    std::cout << "Launching the Kernel" << "\n";
    OCL_CHECK(err, err = q.enqueueTask(krnl_SUNSAL));

    // The result of the previous kernel execution will need to be retrieved in
    // order to view the results. This call will transfer the data from FPGA to
    // source_results vector
    OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_x}, CL_MIGRATE_MEM_OBJECT_HOST));

    OCL_CHECK(err, q.finish());

    /*
    for(int i = 0; i < N; i++) {
    	std::cout << ptr_x[i] << " ";
    }
    std::cout << "\n";
	*/

    // Vector x (N)
	std::cout << "A:\n";
	for(int i = 0; i < 3*3; i++)
		std::cout << ptr_x[i] << " ";
	std::cout << "\n";
	std::cout << "U:\n";
	for(int i = 3*3; i < 2*3*3; i++)
		std::cout << ptr_x[i] << " ";
	std::cout << "\n";
	std::cout << "S:\n";
	for(int i = 2*3*3; i < 3*3*3; i++)
		std::cout << ptr_x[i] << " ";
	std::cout << "\n";
	std::cout << "Ut:\n";
	for(int i = 3*3*3; i < 4*3*3; i++)
		std::cout << ptr_x[i] << " ";
	std::cout << "\n";
	std::cout << "IB:\n";
	for(int i = 4*3*3; i < 5*3*3; i++)
		std::cout << ptr_x[i] << " ";
	std::cout << "\n";
	std::cout << "Solución (x):\n";
	for(int i = 5*3*3; i < 5*3*3 + 3; i++)
			std::cout << ptr_x[i] << " ";
	std::cout << "\n";

    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_A, ptr_A));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_y, ptr_y));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_x, ptr_x));
    OCL_CHECK(err, err = q.finish());

    return EXIT_SUCCESS;

}
