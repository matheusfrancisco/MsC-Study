#include <boost/filesystem.hpp>
#include <vector>
#include <string>
#include <iostream>

namespace fs = boost::filesystem;

std::vector<std::string> list_files(const std::string& directory) {
    std::vector<std::string> files;
    fs::path dir_path(directory);
    if (fs::exists(dir_path) && fs::is_directory(dir_path)) {
        fs::directory_iterator end_iter;
        for (fs::directory_iterator dir_itr(dir_path); dir_itr != end_iter; ++dir_itr) {
            if (fs::is_regular_file(dir_itr->status())) {
                std::string filename = dir_itr->path().filename().string();
                if (!filename.empty() && filename[0] != '.') {
                    files.push_back(filename);
                }
            }
        }
    } else {
        std::cerr << "Directory does not exist or is not accessible: " << directory << std::endl;
    }
    return files;
}


int main() {
    std::string directory = "./dataset/brain_tumor_dataset/Brain-Tumor/";
    std::vector<std::string> files = list_files(directory);

    for (const auto& file : files) {
        std::cout << file << std::endl;
    }

    return 0;
}
