
#include "common.h"
#include "whisper.h"
#include "grammar-parser.h"

#include <string>
#include <thread>
#include <iostream>

using namespace std;

void RunInference(whisper_context* context, int index)
{
    auto input = new float[16000 * 2];
    for (int i = 0; i < 16000 * 2; i++) {
        input[i] = (float)rand() / RAND_MAX * 2 - 1;
    }

    // Generating random wave file with float (1, -1) as input (2 seconds at 16kHz)
    cout << "Staring inference number " << index << endl;
    whisper_state* state = whisper_init_state(context);
    auto params = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
    whisper_full_with_state(context, state, params, input, 16000 * 2);

    cout << "Finished inference number " << index << endl;
    whisper_free_state(state);
}

int main(int argc, char** argv) {
    cout << "Waiting for setting the affinity context on the CPU to make sure only one CPU is used" << endl;

    cin.get();

    whisper_context* ctx = whisper_init_from_file("C:\\Projects\\sandrohanea\\whisper.net\\whisper.cpp\\models\\ggml-tiny.bin");
    RunInference(ctx, 0);

    // Starting some inference calls in parallel
    vector<thread> threads;

    for (int i = 0; i < 4; i++) {
        threads.push_back(thread(RunInference, ctx, i));
    }

    for (auto& t : threads) {
        t.join();
    }

    whisper_free(ctx);
}
