#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char ** argv) {
    if (argc != 10) return 1;
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto elapsed = [&](auto t) { return std::chrono::duration<double>(now()-t).count(); };
    const auto process_start = now();
    const std::string mode = argv[1], state = argv[6], output = argv[7], reference = argv[8], logits_path = argv[9];
    const int n = std::stoi(argv[4]);
    if (n <= 0 || n % 256 || (mode != "seed" && mode != "cold" && mode != "restore" && mode != "restore-evict")) return 2;
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    auto start = now();
    auto model = llama_model_load_from_file(argv[2], mp);
    if (!model || llama_model_n_layer(model) != 32) return 3;
    const double model_s = elapsed(start);
    const auto vocab = llama_model_get_vocab(model);
    auto tokenize = [&](const char * path) {
        std::ifstream file(path);
        std::string text((std::istreambuf_iterator<char>(file)), {});
        if (text.empty()) std::exit(4);
        std::vector<llama_token> tokens(text.size()+16);
        int size = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), false, true);
        if (size <= 0) std::exit(4);
        tokens.resize(size); return tokens;
    };
    auto prefix = tokenize(argv[3]), suffix = tokenize(argv[5]);
    if (int(prefix.size()) < n || suffix.empty() || suffix.size() > 256) return 4;
    prefix.resize(n);
    std::ofstream prompt(output+".prompt.tokens");
    for (auto token : prefix) prompt << token << '\n';
    for (auto token : suffix) prompt << token << '\n';
    prompt.close();
    auto cp = llama_context_default_params();
    cp.n_ctx = n+512; cp.n_batch = cp.n_ubatch = 256;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.type_k = cp.type_v = GGML_TYPE_Q4_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    start = now();
    auto ctx = llama_init_from_model(model, cp);
    if (!ctx) return 5;
    const double context_s = elapsed(start);
    auto batch = llama_batch_init(256, 0, 1);
    auto feed = [&](const llama_token * tokens, int count, int pos, bool logits) {
        batch.n_tokens = count;
        for (int j = 0; j < count; ++j) {
            batch.token[j] = tokens[j]; batch.pos[j] = pos+j;
            batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0;
            batch.logits[j] = logits && j == count-1;
        }
        auto t = now();
        if (llama_decode(ctx, batch)) std::exit(6);
        llama_synchronize(ctx); return elapsed(t);
    };
    double prefix_s = 0, save_s = 0, restore_s = 0;
    const auto request_start = now();
    if (mode == "seed" || mode == "cold") {
        start = now();
        for (int pos = 0; pos < n; pos += 256) {
            feed(prefix.data()+pos, 256, pos, false);
            if ((pos+256)%1024 == 0) std::cerr << "prefix_progress=" << pos+256 << "/" << n << " wall_s=" << elapsed(start) << std::endl;
        }
        prefix_s = elapsed(start);
        if (mode == "seed") {
            if (std::filesystem::exists(state)) return 7;
            start = now();
            if (!llama_state_seq_save_file(ctx, state.c_str(), 0, prefix.data(), prefix.size())) return 7;
            int fd = open(state.c_str(), O_RDWR);
            if (fd < 0 || fsync(fd)) return 7;
            close(fd); save_s = elapsed(start);
        }
    } else {
        if (mode == "restore-evict") {
            int fd = open(state.c_str(), O_RDONLY);
            if (fd < 0 || posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)) return 8;
            close(fd);
        }
        start = now();
        std::vector<llama_token> loaded(n); size_t count = 0;
        if (!llama_state_seq_load_file(ctx, state.c_str(), 0, loaded.data(), loaded.size(), &count)) return 8;
        llama_synchronize(ctx);
        if (count != prefix.size() || loaded != prefix || llama_memory_seq_pos_max(llama_get_memory(ctx), 0) != n-1) return 8;
        restore_s = elapsed(start);
    }
    const double suffix_s = feed(suffix.data(), suffix.size(), n, true);
    const double ready_ttft_s = elapsed(request_start), native_ttft_s = elapsed(process_start);
    const int nv = llama_vocab_n_tokens(vocab);
    std::ofstream token_file(output+".tokens"), text_file(output+".txt"), saved;
    std::ifstream compared;
    if (reference == "-") saved.open(logits_path, std::ios::binary);
    else compared.open(reference, std::ios::binary);
    if (reference == "-" ? !saved : !compared) return 9;
    std::vector<float> expected(nv);
    size_t unequal = 0, checked = 0;
    double max_error = 0, decode_s = 0;
    int generated = 0, decode_calls = 0, pos = n+suffix.size();
    for (int i = 0; i < 32; ++i) {
        const auto logits = llama_get_logits_ith(ctx, -1);
        if (reference == "-") {
            if (!saved.write(reinterpret_cast<const char *>(logits), nv*sizeof(float))) return 9;
        } else {
            if (!compared.read(reinterpret_cast<char *>(expected.data()), nv*sizeof(float))) return 9;
            for (int j = 0; j < nv; ++j) {
                if (std::isnan(logits[j]) || std::isnan(expected[j])) return 9;
                if (std::memcmp(logits+j, expected.data()+j, sizeof(float))) ++unequal;
                if (logits[j] != expected[j]) {
                    if (!std::isfinite(logits[j]) || !std::isfinite(expected[j])) return 9;
                    max_error = std::max(max_error, std::abs(double(logits[j])-expected[j]));
                }
            }
            checked += nv;
        }
        int token = std::max_element(logits, logits+nv)-logits;
        if (llama_vocab_is_eog(vocab, token)) break;
        token_file << token << '\n'; ++generated;
        char piece[512]; int bytes = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
        if (bytes > 0) text_file.write(piece, bytes);
        if (i != 31) { decode_s += feed(&token, 1, pos++, true); ++decode_calls; }
    }
    if (reference != "-" && compared.peek() != std::char_traits<char>::eof()) return 9;
    std::cout << std::setprecision(12) << "{\"prefix_tokens\":" << n << ",\"suffix_tokens\":" << suffix.size()
        << ",\"model_load_s\":" << model_s << ",\"context_init_s\":" << context_s
        << ",\"prefix_s\":" << prefix_s << ",\"save_fsync_s\":" << save_s << ",\"restore_s\":" << restore_s
        << ",\"suffix_s\":" << suffix_s << ",\"model_ready_ttft_s\":" << ready_ttft_s << ",\"native_ttft_s\":" << native_ttft_s
        << ",\"decode_s\":" << decode_s << ",\"decode_calls\":" << decode_calls << ",\"generated_tokens\":" << generated
        << ",\"request_wall_s\":" << elapsed(request_start) << ",\"checkpoint_bytes\":" << std::filesystem::file_size(state)
        << ",\"checked_logits\":" << checked << ",\"bitwise_unequal_logits\":" << unequal << ",\"max_abs_logit_error\":" << max_error << "}" << std::endl;
    llama_batch_free(batch); llama_free(ctx); llama_model_free(model); llama_backend_free();
}
