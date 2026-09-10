#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 8) return 1;
    const int limit = std::stoi(argv[3]), generate = std::stoi(argv[5]), stride = std::stoi(argv[6]);
    const std::string output = argv[4], reference = argv[7];
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    auto model = llama_model_load_from_file(argv[1], mp);
    if (!model || llama_model_n_layer(model) != 32) return 2;
    const auto vocab = llama_model_get_vocab(model);
    std::ifstream input(argv[2]);
    std::string text((std::istreambuf_iterator<char>(input)), {});
    if (text.empty()) return 3;
    std::vector<llama_token> tokens(text.size() + 16);
    int n = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), false, true);
    if (n <= 0 || (limit > 0 && n < limit)) return 4;
    std::vector<llama_token> extra(tokens.begin(), tokens.begin() + std::min(n, 512));
    tokens.resize(limit > 0 ? limit : n);
    n = tokens.size();
    std::ofstream token_file(output + ".prompt.tokens");
    for (auto t : tokens) token_file << t << '\n';
    auto cp = llama_context_default_params();
    cp.n_ctx = n + generate + (std::getenv("BENCH_PROFILE") ? 768 : 256);
    cp.n_batch = cp.n_ubatch = 256;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto ctx = llama_init_from_model(model, cp);
    if (!ctx) return 5;
    auto batch = llama_batch_init(256, 0, 1);
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto elapsed = [&](auto start) { return std::chrono::duration<double>(now() - start).count(); };
    const int nv = llama_vocab_n_tokens(vocab);
    auto prediction = [&](float * logits, int target) {
        int best = std::max_element(logits, logits + nv) - logits;
        double sum = 0;
        for (int j = 0; j < nv; ++j) sum += std::exp(double(logits[j]) - logits[best]);
        double loss = logits[best] + std::log(sum) - logits[target];
        if (!std::isfinite(loss)) std::exit(6);
        return std::pair<int,double>(best, loss);
    };
    double pp = 0, loss_sum = 0;
    int samples = 0;
    std::ofstream scores(output + ".scores.tsv");
    scores << std::setprecision(12);
    auto wall = now();
    for (int offset = 0; offset < n; offset += 256) {
        batch.n_tokens = std::min(256, n - offset);
        for (int j = 0; j < batch.n_tokens; ++j) {
            int pos = offset + j;
            batch.token[j] = tokens[pos]; batch.pos[j] = pos;
            batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0;
            batch.logits[j] = pos == n - 1 || (stride > 0 && pos >= n/2 && pos % stride == 0 && pos+1 < n);
        }
        auto start = now();
        if (llama_decode(ctx, batch)) return 7;
        llama_synchronize(ctx);
        pp += elapsed(start);
        for (int j = 0; j < batch.n_tokens; ++j) {
            int pos = offset + j;
            if (!batch.logits[j] || pos+1 >= n) continue;
            auto [best, loss] = prediction(llama_get_logits_ith(ctx, j), tokens[pos+1]);
            scores << pos << '\t' << tokens[pos+1] << '\t' << best << '\t' << loss << '\n';
            loss_sum += loss; ++samples;
        }
        if ((offset + batch.n_tokens) % 1024 == 0) std::cerr << "prefill_progress=" << offset + batch.n_tokens << "/" << n << " model_s=" << pp << std::endl;
    }
    double pp_wall = elapsed(wall), decode = 0, continuation_loss = 0;
    std::vector<llama_token> forced;
    if (reference != "-") {
        std::ifstream f(reference); int t;
        while (f >> t) forced.push_back(t);
        if (forced.empty()) return 8;
    }
    std::ofstream continuation(output + ".tokens"), response(output + ".txt");
    int steps = 0, matches = 0;
    for (int i = 0; i < generate; ++i) {
        auto logits = llama_get_logits_ith(ctx, -1);
        int best = std::max_element(logits, logits + nv) - logits;
        if (!forced.empty() && i >= int(forced.size())) break;
        int token = forced.empty() ? best : forced[i];
        if (llama_vocab_is_eog(vocab, token)) break;
        continuation_loss += prediction(logits, token).second;
        matches += token == best;
        continuation << token << '\n';
        char piece[512]; int size = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
        if (size > 0) response.write(piece, size);
        batch.n_tokens = 1; batch.token[0] = token; batch.pos[0] = n+i;
        batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
        auto start = now();
        if (llama_decode(ctx, batch)) return 9;
        llama_synchronize(ctx); decode += elapsed(start); ++steps;
    }
    std::cout << std::setprecision(12) << "{\"prompt_tokens\":" << n << ",\"prefill_s\":" << pp
        << ",\"prefill_wall_s\":" << pp_wall << ",\"samples\":" << samples << ",\"nll\":" << (samples ? loss_sum/samples : 0)
        << ",\"decode_steps\":" << steps << ",\"decode_s\":" << decode << ",\"decode_tps\":" << (steps ? steps/decode : 0)
        << ",\"continuation_nll\":" << (steps ? continuation_loss/steps : 0) << ",\"continuation_matches\":" << matches << "}" << std::endl;
    if (std::getenv("BENCH_PROFILE")) {
        int pos = n + steps;
        const int pad = (256 - pos % 256) % 256;
        auto append = [&](int count) {
            batch.n_tokens = count;
            for (int j = 0; j < count; ++j) {
                batch.token[j] = extra[j % extra.size()]; batch.pos[j] = pos+j;
                batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0; batch.logits[j] = j == count-1;
            }
            if (llama_decode(ctx, batch)) std::exit(10);
            llama_synchronize(ctx); pos += count;
        };
        if (pad) append(pad);
        std::cerr << "profile_begin position=" << pos << " tokens=256" << std::endl;
        setenv("GGML_VK_PROFILE_NOW", "1", 1);
        auto start = now(); append(256);
        std::cerr << "profile_end wall_s=" << elapsed(start) << std::endl;
    }
    llama_batch_free(batch); llama_free(ctx); llama_model_free(model); llama_backend_free();
}
