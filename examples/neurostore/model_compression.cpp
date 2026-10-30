
/**
 * @file model_compression.cpp
 * Demonstrate model compression using NeurStore
 */

// make sure you have local neurostore runtime enviroment
#include "neurstore/neurstore.h"

#include <filesystem>
#include <chrono>
#include <iostream>

namespace fs = std::filesystem;
using namespace std::chrono;

std::vector<uint8_t> read_file(const std::string &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("File is empty or unreadable: " + path);
    }
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read file contents: " + path);
    }
    return buffer;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model_folder> <tolerance> <parallelism>" << std::endl;
        return 1;
    }
    const std::string model_folder = argv[1];
    const double tolerance = std::stod(argv[2]);
    const int parallelism = std::stoi(argv[3]);

    omp_set_num_threads(parallelism);

    std::vector<std::string> model_names;

    for (const auto& entry : fs::directory_iterator(model_folder)) {
        if (entry.is_regular_file()) {
            const std::string filename = entry.path().filename().string();
            if (entry.path().extension() == ".onnx") {
                model_names.push_back(filename.substr(0, filename.size() - 5));
            }
        }
    }

    std::string outputFolder = "./neurstore_compression_example";
    fs::create_directories(outputFolder);

    const NeurStore neurstore(outputFolder, std::make_shared<IndexCacheManager>(outputFolder));
    auto start = high_resolution_clock::now();
    bool success = false;
    try {
        success = neurstore.saveModels(model_names, tolerance, model_folder);
    } catch (const std::exception &e) {
        std::cerr << "Exception during compression: " << e.what() << std::endl;
        return 1;
    }
    if (!success) {
        std::cerr << "Compression failed." << std::endl;
        return 1;
    }
    const auto end = high_resolution_clock::now();
    const double elapsed = duration_cast<duration<double>>(end - start).count();

    double compressed_size = 0.0; // MB
    for (const auto &file : std::filesystem::recursive_directory_iterator(outputFolder)) {
        if (file.is_regular_file()) {
            compressed_size += static_cast<double>(file.file_size()) / (1024 * 1024);
        }
    }

    double original_size = 0.0; // MB
    for (const auto &name: model_names) {
        std::string path = fs::path(model_folder) / (name + ".onnx");
        try {
            auto raw = read_file(path);
            original_size += static_cast<double>(raw.size()) / (1024 * 1024);
        } catch (...) {
            std::cerr << "Failed to read: " << path << std::endl;
        }
    }

    std::cout << "================ Compression Summary ================" << std::endl;
    std::cout << "Compressing models from folder: " << model_folder << std::endl;
    std::cout << "Tolerance: " << tolerance << ", Parallelism: " << parallelism << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << "Total models compressed: " << model_names.size() << std::endl;
    std::cout << "Total time: " << elapsed << " sec" << std::endl;
    std::cout << "Total original size: " << original_size << " MB" << std::endl;
    std::cout << "Total compressed size: " << compressed_size << " MB" << std::endl;
    std::cout << "Compression ratio: " << (original_size / compressed_size) << std::endl;
    std::cout << "=====================================================" << std::endl;
    return 0;
}
