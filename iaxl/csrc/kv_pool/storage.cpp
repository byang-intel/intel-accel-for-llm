// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "env.h"
#include "kv_pool.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>
#include <utility>

namespace kv_pool {

class Storage::Impl {
  public:
    explicit Impl(const std::string &persist_dir) : persist_dir_(persist_dir) {
        if (persist_dir_.empty()) {
            std::cerr << "\033[1;31m*** FATAL: persist_dir is not set, cannot save data\033[0m\n";
            std::abort();
        }
        if (envs.IAXL_DEBUG_LOG && !persist_dir_.empty()) {
            std::cout << "[storage] Initialized with persist_dir: " << persist_dir_ << "\n";
        }
    }

    bool save(const std::string &full_label, const char *data, size_t size) {

        std::string path_str = get_path_from_label(full_label);

        std::filesystem::path file_path =
            std::filesystem::path(persist_dir_) / "chunks" / (path_str + ".bin");
        std::filesystem::path dir_path = file_path.parent_path();

        std::error_code ec;
        std::filesystem::create_directories(dir_path, ec);
        if (ec) {
            std::cerr << "\033[1;31m*** FATAL: error in creating directory " << dir_path << ": "
                      << ec.message() << "\033[0m\n"
                      << std::endl;
            std::abort();
        }

        std::ofstream file(file_path, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            std::cerr << "\033[1;31m*** FATAL: Error opening file for writing: " << file_path
                      << "\033[0m\n"
                      << std::endl;
            std::abort();
        }

        file.write(data, size);
        if (file.tellp() != static_cast<std::streampos>(size)) {
            std::cerr << "\033[1;31m*** FATAL: Error writing file: " << file_path << ", wrote "
                      << file.tellp() << " bytes, expected " << size << "\033[0m\n"
                      << std::endl;
            std::abort();
        }
        file.close();

        if (envs.IAXL_DEBUG_LOG) {
            std::cout << "[storage] Saved " << size << " bytes to " << file_path << "\n";
        }

        return true;
    }

    std::pair<char *, size_t> load(const std::string &full_label) {

        std::string path_str = get_path_from_label(full_label);

        std::filesystem::path file_path =
            std::filesystem::path(persist_dir_) / "chunks" / (path_str + ".bin");

        if (envs.IAXL_DEBUG_LOG) {
            std::cout << "[storage] Attempting to load from file: " << file_path << "\n";
        }

        if (!std::filesystem::exists(file_path)) {
            std::cerr << "\033[1;31m*** FATAL: File not found: " << file_path << "\033[0m\n"
                      << std::endl;
            std::abort();
        }

        std::ifstream file(file_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "\033[1;31m*** FATAL: Error opening file for reading: " << file_path
                      << "\033[0m\n"
                      << std::endl;
            std::abort();
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        char *buffer = (char *)malloc(size);
        if (!buffer) {
            std::cerr << "\033[1;31m*** FATAL: Memory allocation failed for size: " << size
                      << "\033[0m\n"
                      << std::endl;
            std::abort();
        }

        if (file.read(buffer, size)) {
            if (file.gcount() != size) {
                free(buffer);
                std::cerr << "\033[1;31m*** FATAL: Error reading file: " << file_path << ", read "
                          << file.gcount() << " bytes, expected " << size << "\033[0m\n"
                          << std::endl;
                std::abort();
            }
            return {buffer, static_cast<size_t>(size)};
        } else {
            free(buffer);
            std::cerr << "\033[1;31m*** FATAL: Error reading file: " << file_path << "\033[0m\n"
                      << std::endl;
            std::abort();
        }
    }

  private:
    std::string persist_dir_;
};

Storage::Storage(const std::string &persist_dir) : impl_(std::make_unique<Impl>(persist_dir)) {}
Storage::~Storage() = default;

bool Storage::save(const std::string &full_label, const char *data, size_t size) {
    return impl_->save(full_label, data, size);
}
std::pair<char *, size_t> Storage::load(const std::string &full_label) {
    return impl_->load(full_label);
}

} // namespace kv_pool
