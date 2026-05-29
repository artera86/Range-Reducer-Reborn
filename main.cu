// author: https://t.me/biernus
// Modified: Single random base + thread offset approach
#include "secp256k1.cuh"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <cstdint>
#include <fstream>
#include <stdint.h>
#include <curand_kernel.h>
#include <algorithm>
#include <random>
#include <inttypes.h>
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#include <chrono>
#pragma once

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "Error: CUDA failure in " << #call \
                  << " (" << cudaGetErrorString(err) << ")" << std::endl; \
        return false; \
    } \
} while(0)

static bool is_valid_hex(const char* str) {
    if (!str || *str == '\0') return false;
    for (int i = 0; str[i] != '\0'; i++) {
        char c = str[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')))
            return false;
    }
    return true;
}

static bool hex_bigint_le(const char* a, const char* b) {
    size_t len_a = strlen(a);
    size_t len_b = strlen(b);
    if (len_a != len_b) return len_a < len_b;
    for (size_t i = 0; i < len_a; i++) {
        char ca = (a[i] >= 'A' && a[i] <= 'F') ? (a[i] - 'A' + 'a') : a[i];
        char cb = (b[i] >= 'A' && b[i] <= 'F') ? (b[i] - 'A' + 'a') : b[i];
        if (ca < cb) return true;
        if (ca > cb) return false;
    }
    return true;
}

__device__ __host__ __forceinline__ uint8_t hex_char_to_byte(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

// Convert hex string to bytes
__device__ __host__ void hex_string_to_bytes(const char* hex_str, uint8_t* bytes, int num_bytes) {
    #pragma unroll 8
    for (int i = 0; i < num_bytes; i++) {
        bytes[i] = (hex_char_to_byte(hex_str[i * 2]) << 4) | 
                   hex_char_to_byte(hex_str[i * 2 + 1]);
    }
}

// Convert hex string to BigInt - optimized
__device__ __host__ void hex_to_bigint(const char* hex_str, BigInt* bigint) {
    // Initialize all data to 0
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        bigint->data[i] = 0;
    }
    
    int len = 0;
    while (hex_str[len] != '\0' && len < 64) len++;
    
    // Process hex string from right to left
    int word_idx = 0;
    int bit_offset = 0;
    
    for (int i = len - 1; i >= 0 && word_idx < 8; i--) {
        uint8_t val = hex_char_to_byte(hex_str[i]);
        
        bigint->data[word_idx] |= ((uint32_t)val << bit_offset);
        
        bit_offset += 4;
        if (bit_offset >= 32) {
            bit_offset = 0;
            word_idx++;
        }
    }
}

// Convert BigInt to hex string - optimized
__device__ void bigint_to_hex(const BigInt* bigint, char* hex_str) {
    const char hex_chars[] = "0123456789abcdef";
    int idx = 0;
    bool leading_zero = true;
    
    // Process from most significant word to least
    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        for (int j = 28; j >= 0; j -= 4) {
            uint8_t nibble = (bigint->data[i] >> j) & 0xF;
            if (nibble != 0 || !leading_zero || (i == 0 && j == 0)) {
                hex_str[idx++] = hex_chars[nibble];
                leading_zero = false;
            }
        }
    }
    
    // Handle case where number is 0
    if (idx == 0) {
        hex_str[idx++] = '0';
    }
    
    hex_str[idx] = '\0';
}

// Optimized byte to hex conversion
__device__ __forceinline__ void byte_to_hex(uint8_t byte, char* out) {
    const char hex_chars[] = "0123456789abcdef";
    out[0] = hex_chars[(byte >> 4) & 0xF];
    out[1] = hex_chars[byte & 0xF];
}

__device__ void hash160_to_hex(uint8_t* hash, char* hex_str) {
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        byte_to_hex(hash[i], &hex_str[i * 2]);
    }
    hex_str[40] = '\0';
}

__device__ __forceinline__ bool compare_hash160_fast(const uint8_t* hash1, const uint8_t* hash2) {
    uint64_t a1, a2, b1, b2;
    uint32_t c1, c2;
    
    memcpy(&a1, hash1, 8);
    memcpy(&a2, hash1 + 8, 8);
    memcpy(&c1, hash1 + 16, 4);

    memcpy(&b1, hash2, 8);
    memcpy(&b2, hash2 + 8, 8);
    memcpy(&c2, hash2 + 16, 4);

    return (a1 == b1) && (a2 == b2) && (c1 == c2);
}

__device__ void hash160_to_hex(const uint8_t *hash, char *out_hex) {
    const char hex_chars[] = "0123456789abcdef";
    for (int i = 0; i < 20; ++i) {
        out_hex[i * 2]     = hex_chars[hash[i] >> 4];
        out_hex[i * 2 + 1] = hex_chars[hash[i] & 0x0F];
    }
    out_hex[40] = '\0';
}

// Device function to generate random BigInt in range [min, max]
__device__ void generate_random_in_range(BigInt* result, curandStateMRG32k3a_t* state, 
                                         const BigInt* min_val, const BigInt* max_val) {
    // Calculate range = max - min
    BigInt range;
    bool borrow = false;
    
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        uint64_t diff = (uint64_t)max_val->data[i] - (uint64_t)min_val->data[i] - (borrow ? 1 : 0);
        range.data[i] = (uint32_t)diff;
        borrow = (diff > 0xFFFFFFFFULL);
    }
    
    // Generate random value in [0, range]
    BigInt random;
    for (int w = 0; w < BIGINT_WORDS; w += 4) {
        if (w + 0 < BIGINT_WORDS) random.data[w + 0] = curand(state);
        if (w + 1 < BIGINT_WORDS) random.data[w + 1] = curand(state);
        if (w + 2 < BIGINT_WORDS) random.data[w + 2] = curand(state);
        if (w + 3 < BIGINT_WORDS) random.data[w + 3] = curand(state);
    }
    
    // Reduce random to range
    int highest_word = BIGINT_WORDS - 1;
    while (highest_word >= 0 && range.data[highest_word] == 0) {
        highest_word--;
    }
    
    if (highest_word >= 0) {
        uint32_t mask = range.data[highest_word];
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        
        random.data[highest_word] &= mask;
        
        for (int i = highest_word + 1; i < BIGINT_WORDS; ++i) {
            random.data[i] = 0;
        }
        
        bool greater = false;
        for (int i = BIGINT_WORDS - 1; i >= 0; --i) {
            if (random.data[i] > range.data[i]) {
                greater = true;
                break;
            } else if (random.data[i] < range.data[i]) {
                break;
            }
        }
        
        if (greater) {
            for (int i = 0; i < BIGINT_WORDS; ++i) {
                random.data[i] = random.data[i] % (range.data[i] + 1);
            }
        }
    }
    
    // Add min: result = random + min
    bool carry = false;
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        uint64_t sum = (uint64_t)random.data[i] + (uint64_t)min_val->data[i] + (carry ? 1 : 0);
        result->data[i] = (uint32_t)sum;
        carry = (sum > 0xFFFFFFFFULL);
    }
}
__device__ __host__ void clear_last_6_hex(BigInt* num) {
    // Last 5 hex digits = 20 bits
    // Clear the least significant 20 bits
    num->data[0] &= 0xFF000000;  // Keep upper 12 bits, clear lower 20 bits
}
// Global device constants for min/max as BigInt
__constant__ BigInt d_min_bigint;
__constant__ BigInt d_max_bigint;
__constant__ BigInt d_base_key;  // Shared base key for all threads

__device__ volatile int g_found = 0;
__device__ char g_found_hex[65] = {0};
__device__ char g_found_hash160[41] = {0};

// OPTIMIZED KERNEL: Each thread checks BATCH_SIZE sequential keys
// Starting from: base_key + (tid * BATCH_SIZE)
__global__ void start(const uint8_t* target, int total_threads)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate this thread's starting private key: base_key + (tid * BATCH_SIZE)
    BigInt priv_base;
    BigInt offset;
    init_bigint(&offset, tid * BATCH_SIZE);
    
    // priv_base = (base_key + tid * BATCH_SIZE) mod n
    ptx_u256Add(&priv_base, &d_base_key, &offset);
    
    // Reduce modulo n if needed
    if (compare_bigint(&priv_base, &const_n) >= 0) {
        ptx_u256Sub(&priv_base, &priv_base, &const_n);
    }
    
    // Check if starting key is within [min, max] range
    if (compare_bigint(&priv_base, &d_min_bigint) < 0 || 
        compare_bigint(&priv_base, &d_max_bigint) > 0) {
        return; // Skip if out of range
    }
    
    // Array to hold batch of points
    ECPointJac result_jac_batch[BATCH_SIZE];
    uint8_t hash160_batch[BATCH_SIZE][20];
    //clear_last_5_hex(&priv_base);
    // --- Compute base point: P = priv_base * G ---
    scalar_multiply_multi_base_jac(&result_jac_batch[0], &priv_base);
    
    // --- Generate sequential points: P+G, P+2G, P+3G, ... P+(BATCH_SIZE-1)G ---
    #pragma unroll
    for (int i = 1; i < BATCH_SIZE; ++i) {
        // result_jac_batch[i] = result_jac_batch[i-1] + G
        add_G_to_point_jac(&result_jac_batch[i], &result_jac_batch[i-1]);
    }
    
    // --- Convert the entire batch to hash160s with ONE inverse ---
    jacobian_batch_to_hash160(result_jac_batch, hash160_batch);
    
    // Debug print for first thread
    if (tid == 0) {
        char hash160_str[41];
        char hex_key[65];
        bigint_to_hex(&priv_base, hex_key);
        hash160_to_hex(hash160_batch[0], hash160_str);
        printf("Base key: %s -> %s (checking %d sequential keys)\n", hex_key, hash160_str, BATCH_SIZE);
    }
    
    // --- Check all results from the batch ---
    #pragma unroll
    for (int i = 0; i < BATCH_SIZE; ++i) {
        if (compare_hash160_fast(hash160_batch[i], target)) {
            if (atomicCAS((int*)&g_found, 0, 1) == 0) {
                // Calculate the actual private key: priv_base + i
                BigInt priv_found;
                BigInt key_offset;
                init_bigint(&key_offset, i);
                
                // priv_found = (priv_base + i) mod n
                ptx_u256Add(&priv_found, &priv_base, &key_offset);
                
                // Reduce modulo n if needed
                if (compare_bigint(&priv_found, &const_n) >= 0) {
                    ptx_u256Sub(&priv_found, &priv_found, &const_n);
                }
                
                char found_hex[65];
                bigint_to_hex(&priv_found, found_hex);
                hash160_to_hex(hash160_batch[i], g_found_hash160);
                memcpy(g_found_hex, found_hex, 65);
                return;
            }
        }
    }
}

bool run_with_quantum_data(const char* min, const char* max, const char* target, int blocks, int threads, int device_id) {
    cudaError_t dev_err = cudaSetDevice(device_id);
    if (dev_err != cudaSuccess) {
        std::cerr << "Error: Failed to set CUDA device " << device_id
                  << " (" << cudaGetErrorString(dev_err) << ")" << std::endl;
        return false;
    }

    uint8_t shared_target[20];
    hex_string_to_bytes(target, shared_target, 20);
    uint8_t *d_target;
    CUDA_CHECK(cudaMalloc(&d_target, 20));
    CUDA_CHECK(cudaMemcpy(d_target, shared_target, 20, cudaMemcpyHostToDevice));
    
    // Convert min and max hex strings to BigInt and copy to device
    BigInt min_bigint, max_bigint;
    hex_to_bigint(min, &min_bigint);
    hex_to_bigint(max, &max_bigint);
    
    CUDA_CHECK(cudaMemcpyToSymbol(d_min_bigint, &min_bigint, sizeof(BigInt)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_max_bigint, &max_bigint, sizeof(BigInt)));
    
    int total_threads = blocks * threads;
    int found_flag;
    
    // Calculate keys processed per kernel launch
    uint64_t keys_per_kernel = (uint64_t)total_threads * BATCH_SIZE;
    
    printf("Searching in range:\n");
    printf("Min: %s\n", min);
    printf("Max: %s\n", max);
    printf("Target: %s\n", target);
    printf("Blocks: %d, Threads: %d, Batch size: %d\n", blocks, threads, BATCH_SIZE);
    printf("Total threads: %d\n", total_threads);
    printf("Keys per kernel: %llu (each thread checks %d sequential keys)\n\n", 
           (unsigned long long)keys_per_kernel, BATCH_SIZE);
    
    // Performance tracking variables
    uint64_t total_keys_checked = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    //auto last_print_time = start_time;
    int iteration = 0;
    
    // Setup random number generator for host
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, 0xFFFFFFFF);
    
    while(true) {
        // Generate ONE random base key for all threads on HOST
        BigInt base_key;
        
        // Calculate range = max - min
        BigInt range;
        bool borrow = false;
        for (int i = 0; i < BIGINT_WORDS; ++i) {
            uint64_t diff = (uint64_t)max_bigint.data[i] - (uint64_t)min_bigint.data[i] - (borrow ? 1 : 0);
            range.data[i] = (uint32_t)diff;
            borrow = (diff > 0xFFFFFFFFULL);
        }
        
        // Generate random value in [0, range]
        BigInt random_val;
        for (int i = 0; i < BIGINT_WORDS; ++i) {
            random_val.data[i] = dis(gen);
        }
        
        // Simple reduction to fit within range
        int highest_word = BIGINT_WORDS - 1;
        while (highest_word >= 0 && range.data[highest_word] == 0) {
            highest_word--;
        }
        
        if (highest_word >= 0) {
            uint32_t mask = range.data[highest_word];
            mask |= mask >> 1;
            mask |= mask >> 2;
            mask |= mask >> 4;
            mask |= mask >> 8;
            mask |= mask >> 16;
            
            random_val.data[highest_word] &= mask;
            
            for (int i = highest_word + 1; i < BIGINT_WORDS; ++i) {
                random_val.data[i] = 0;
            }
        }
        
        // base_key = random_val + min
        bool carry = false;
        for (int i = 0; i < BIGINT_WORDS; ++i) {
            uint64_t sum = (uint64_t)random_val.data[i] + (uint64_t)min_bigint.data[i] + (carry ? 1 : 0);
            base_key.data[i] = (uint32_t)sum;
            carry = (sum > 0xFFFFFFFFULL);
        }
		
		clear_last_6_hex(&base_key);
        
        // Copy base key to device constant memory
        cudaError_t sym_err = cudaMemcpyToSymbol(d_base_key, &base_key, sizeof(BigInt));
        if (sym_err != cudaSuccess) {
            std::cerr << "Error: Failed to copy base key to device"
                      << " (" << cudaGetErrorString(sym_err) << ")" << std::endl;
            cudaFree(d_target);
            return false;
        }
        
        auto kernel_start = std::chrono::high_resolution_clock::now();
        
        // Launch kernel - each thread checks base_key + thread_id
        start<<<blocks, threads>>>(d_target, total_threads);
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            std::cerr << "Error: Kernel launch failed"
                      << " (" << cudaGetErrorString(launch_err) << ")" << std::endl;
            cudaFree(d_target);
            return false;
        }
        cudaDeviceSynchronize();
        
        auto kernel_end = std::chrono::high_resolution_clock::now();
        
        // Calculate kernel execution time
        double kernel_time = std::chrono::duration<double>(kernel_end - kernel_start).count();
        
        // Update counters (each kernel launch checks total_threads * BATCH_SIZE keys)
        uint64_t keys_per_kernel = (uint64_t)total_threads * BATCH_SIZE;
        total_keys_checked += keys_per_kernel;
        iteration++;
        /*
        // Print performance stats every second
        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed_since_print = std::chrono::duration<double>(current_time - last_print_time).count();
        
        if (elapsed_since_print >= 1.0 || iteration % 100 == 0) {
            double total_elapsed = std::chrono::duration<double>(current_time - start_time).count();
            double current_kps = keys_per_kernel / kernel_time;
            double average_kps = total_keys_checked / total_elapsed;
            
            printf("\r[Iter %d] Kernel: %.3f ms | Current: %.2f MK/s | Average: %.2f MK/s | Total: %.2f M keys",
                   iteration,
                   kernel_time * 1000.0,
                   current_kps / 1000000.0,
                   average_kps / 1000000.0,
                   total_keys_checked / 1000000.0);
            fflush(stdout);
            
            last_print_time = current_time;
        }
        */
        // Check if key was found
        cudaMemcpyFromSymbol(&found_flag, g_found, sizeof(int));
        if (found_flag) {
            printf("\n\n");
            
            char found_hex[65], found_hash160[41];
            cudaMemcpyFromSymbol(found_hex, g_found_hex, 65);
            cudaMemcpyFromSymbol(found_hash160, g_found_hash160, 41);
            
            double total_time = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - start_time
            ).count();
            
            printf("FOUND!\n");
            printf("Private Key: %s\n", found_hex);
            printf("Hash160: %s\n", found_hash160);
            printf("Total time: %.2f seconds\n", total_time);
            printf("Total keys checked: %llu (%.2f million)\n", 
                   (unsigned long long)total_keys_checked,
                   total_keys_checked / 1000000.0);
            printf("Average speed: %.2f MK/s\n", total_keys_checked / total_time / 1000000.0);
            
            std::ofstream outfile("result.txt", std::ios::app);
            if (outfile.is_open()) {
                std::time_t now = std::time(nullptr);
                char timestamp[100];
                std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
                outfile << "[" << timestamp << "] Found: " << found_hex << " -> " << found_hash160 << std::endl;
                outfile << "Total keys checked: " << total_keys_checked << std::endl;
                outfile << "Time taken: " << total_time << " seconds" << std::endl;
                outfile << "Average speed: " << (total_keys_checked / total_time / 1000000.0) << " MK/s" << std::endl;
                outfile << std::endl;
                outfile.close();
                std::cout << "Result appended to result.txt" << std::endl;
            }
            
            cudaFree(d_target);
            return true;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <min> <max> <target_hash160> [blocks] [threads] [device_id]" << std::endl;
        std::cerr << "  <min>            Start of hex range (e.g. 100000)" << std::endl;
        std::cerr << "  <max>            End of hex range   (e.g. 1fffff)" << std::endl;
        std::cerr << "  <target_hash160> 40-char hex Hash160 to find" << std::endl;
        std::cerr << "  [blocks]         CUDA blocks   (default: 1024)" << std::endl;
        std::cerr << "  [threads]        CUDA threads  (default: 128)" << std::endl;
        std::cerr << "  [device_id]      GPU device ID (default: 0)" << std::endl;
        return 1;
    }

    const char* min_hex    = argv[1];
    const char* max_hex    = argv[2];
    const char* target_hex = argv[3];

    // --- Validate hex strings ---
    if (!is_valid_hex(min_hex)) {
        std::cerr << "Error: <min> contains invalid hex characters: " << min_hex << std::endl;
        return 1;
    }
    if (!is_valid_hex(max_hex)) {
        std::cerr << "Error: <max> contains invalid hex characters: " << max_hex << std::endl;
        return 1;
    }
    if (!is_valid_hex(target_hex)) {
        std::cerr << "Error: <target_hash160> contains invalid hex characters: " << target_hex << std::endl;
        return 1;
    }

    // --- Validate lengths ---
    size_t min_len    = strlen(min_hex);
    size_t max_len    = strlen(max_hex);
    size_t target_len = strlen(target_hex);

    if (min_len > 64) {
        std::cerr << "Error: <min> exceeds 64 hex characters (256-bit limit)" << std::endl;
        return 1;
    }
    if (max_len > 64) {
        std::cerr << "Error: <max> exceeds 64 hex characters (256-bit limit)" << std::endl;
        return 1;
    }
    if (min_len != max_len) {
        std::cerr << "Error: <min> and <max> must have the same hex length (got "
                  << min_len << " vs " << max_len << ")" << std::endl;
        return 1;
    }
    if (target_len != 40) {
        std::cerr << "Error: <target_hash160> must be exactly 40 hex characters (got "
                  << target_len << ")" << std::endl;
        return 1;
    }

    // --- Validate min <= max ---
    if (!hex_bigint_le(min_hex, max_hex)) {
        std::cerr << "Error: <min> (" << min_hex << ") is greater than <max> ("
                  << max_hex << ")" << std::endl;
        return 1;
    }

    // --- Parse optional numeric arguments ---
    int blocks = 1024;
    int threads = 128;
    int device_id = 0;

    auto parse_positive_int = [](const char* str, const char* name, int& out) -> bool {
        try {
            size_t pos = 0;
            int val = std::stoi(str, &pos);
            if (pos != strlen(str)) {
                std::cerr << "Error: [" << name << "] is not a valid integer: " << str << std::endl;
                return false;
            }
            if (val <= 0) {
                std::cerr << "Error: [" << name << "] must be a positive integer (got "
                          << val << ")" << std::endl;
                return false;
            }
            out = val;
            return true;
        } catch (const std::invalid_argument&) {
            std::cerr << "Error: [" << name << "] is not a valid integer: " << str << std::endl;
            return false;
        } catch (const std::out_of_range&) {
            std::cerr << "Error: [" << name << "] value out of range: " << str << std::endl;
            return false;
        }
    };

    if (argc >= 5 && !parse_positive_int(argv[4], "blocks", blocks)) return 1;
    if (argc >= 6 && !parse_positive_int(argv[5], "threads", threads)) return 1;
    if (argc >= 7) {
        try {
            size_t pos = 0;
            device_id = std::stoi(argv[6], &pos);
            if (pos != strlen(argv[6])) {
                std::cerr << "Error: [device_id] is not a valid integer: " << argv[6] << std::endl;
                return 1;
            }
            if (device_id < 0) {
                std::cerr << "Error: [device_id] must be non-negative (got " << device_id << ")" << std::endl;
                return 1;
            }
        } catch (const std::invalid_argument&) {
            std::cerr << "Error: [device_id] is not a valid integer: " << argv[6] << std::endl;
            return 1;
        } catch (const std::out_of_range&) {
            std::cerr << "Error: [device_id] value out of range: " << argv[6] << std::endl;
            return 1;
        }
    }

    // --- Validate CUDA device ---
    int device_count = 0;
    cudaError_t count_err = cudaGetDeviceCount(&device_count);
    if (count_err != cudaSuccess) {
        std::cerr << "Error: Failed to query CUDA devices (" << cudaGetErrorString(count_err) << ")" << std::endl;
        return 1;
    }
    if (device_count == 0) {
        std::cerr << "Error: No CUDA-capable GPU detected" << std::endl;
        return 1;
    }
    if (device_id >= device_count) {
        std::cerr << "Error: device_id " << device_id << " is invalid (" << device_count
                  << " device(s) available, valid range 0-" << device_count - 1 << ")" << std::endl;
        return 1;
    }

    init_gpu_constants();
    cudaDeviceSynchronize();

    bool result = run_with_quantum_data(min_hex, max_hex, target_hex, blocks, threads, device_id);
    if (!result) {
        std::cerr << "Error: Search did not complete successfully" << std::endl;
        return 1;
    }

    return 0;
}