#include "ggml.h"
#include "ggml-backend.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 5) return 1;
    const int n = std::stoi(argv[1]);
    if (n <= 0) return 2;
    ggml_backend_load_all();
    auto device = ggml_backend_dev_by_name("Vulkan0");
    if (!device) return 3;
    auto backend = ggml_backend_dev_init(device, nullptr);
    auto ctx = ggml_init({ggml_tensor_overhead()*32 + ggml_graph_overhead(), nullptr, true});
    auto x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n, 256, 16);
    auto y = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n, 256, 16);
    auto gram = ggml_mul_mat(ctx, x, x);
    auto cross = ggml_mul_mat(ctx, x, y);
    ggml_prec_set_acc(gram, GGML_PREC_F32);
    ggml_prec_set_acc(cross, GGML_PREC_F32);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, gram);
    ggml_build_forward_expand(graph, cross);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    for (int i = 0; i < 2; ++i) {
        auto tensor = i ? y : x;
        std::vector<float> values(ggml_nelements(tensor));
        std::ifstream input(argv[2+i], std::ios::binary);
        if (!input.read(reinterpret_cast<char *>(values.data()), ggml_nbytes(tensor))) return 4;
        ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
    }
    auto start = std::chrono::steady_clock::now();
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 5;
    ggml_backend_synchronize(backend);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::ofstream output(argv[4], std::ios::binary);
    for (auto tensor : {gram, cross}) {
        std::vector<float> values(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
        if (!output.write(reinterpret_cast<char *>(values.data()), ggml_nbytes(tensor))) return 6;
    }
    std::cout << "{\"backend\":\"" << ggml_backend_dev_name(device) << "\",\"covariance_s\":" << seconds << "}" << std::endl;
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}
