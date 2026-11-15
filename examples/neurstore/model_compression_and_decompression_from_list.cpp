#include "neurstore/neurstore.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <omp.h>

namespace fs = std::filesystem;

std::vector<std::string> readModelList(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Failed to open model list file: " + path);

    std::vector<std::string> modelPaths;
    std::string line;
    while (std::getline(file, line))
        if (!line.empty()) modelPaths.emplace_back(std::move(line));
    return modelPaths;
}

std::vector<char> readFile(const std::string &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Failed to open file: " + path);

    std::streamsize size = file.tellg();
    if (size <= 0)
        throw std::runtime_error("Empty or invalid file: " + path);

    std::vector<char> buffer(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(buffer.data(), size))
        throw std::runtime_error("Failed to read file: " + path);
    return buffer;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                << " <model_list.txt> <store_dir> <tolerance> <parallelism> <load_model_dir>\n";
        return 1;
    }

    const std::string modelListPath = argv[1];
    std::string storeDir = argv[2];
    const double tolerance = std::stod(argv[3]);
    const int parallelism = std::stoi(argv[4]);
    const std::string loadModelDir = argv[5];

    omp_set_num_threads(parallelism);
    fs::create_directories(storeDir);
    fs::create_directories(loadModelDir);

    const auto modelPaths = readModelList(modelListPath);
    auto cacheManager = std::make_shared<IndexCacheManager>(storeDir);
    NeurStore neurstore(storeDir, cacheManager);

    try {
        if (!neurstore.saveModelsFromPaths(modelPaths, tolerance)) {
            std::cerr << "Failed to save models to NeurStore.\n";
            return 1;
        }
        cacheManager->clearCache();
    } catch (const std::exception &e) {
        std::cerr << "Compression exception: " << e.what() << '\n';
        return 1;
    }

    for (const auto &modelPath: modelPaths) {
        fs::path p(modelPath);
        const std::string modelName = p.stem().string();

        try {
            auto modelProto = neurstore.loadModel(modelName);
            std::string buffer;
            modelProto->SerializeToString(&buffer);

            fs::path outPath = fs::path(loadModelDir) / (modelName + ".onnx");
            std::ofstream fout(outPath, std::ios::binary);
            fout.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            std::cout << "[INFO] Loaded & dumped model: " << modelName << " -> "
                    << outPath << '\n';
        } catch (const std::exception &e) {
            std::cerr << "[WARN] Failed to load/dump model " << modelName
                    << ": " << e.what() << '\n';
        }
    }

    double totalOriginalSize = 0.0, totalCompressedSize = 0.0;
    for (const auto &mp: modelPaths)
        totalOriginalSize += static_cast<double>(readFile(mp).size()) / (1024 * 1024);

    for (const auto &entry: fs::directory_iterator(storeDir))
        if (entry.is_regular_file())
            totalCompressedSize += static_cast<double>(entry.file_size()) / (1024 * 1024);

    std::cout << "\n================ Overall Summary (NeurStore) ================\n";
    std::cout << "Total models processed   : " << modelPaths.size() << '\n';
    std::cout << "Compression ratio        : " << (totalOriginalSize / totalCompressedSize) << '\n';
    std::cout << "Total original size      : " << totalOriginalSize << " MB\n";
    std::cout << "Total compressed size    : " << totalCompressedSize << " MB\n";
    std::cout << "=============================================================\n";
    return 0;
}
