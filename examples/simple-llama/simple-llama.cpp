// commit: af0a5b616359809ce886ea433acedebb39b12969
#include "common.h"

#include "simple-llama.h"

#include "console.h"
#include "llama.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static size_t utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

int main(int argc, char ** argv) {
    std::string model_filename, prompt;

    // ELYZA
    {
        model_filename = "/tmp/test/ELYZA-japanese-Llama-2-7b-fast-q4_K_M.gguf";
        prompt = "[INST] <<SYS>>\n"
            "あなたは誠実で優秀な日本人のアシスタントです。\n"
            "<</SYS>>\n\n"
            " こんにちは。 [/INST]"
            "  こんにちは、どういうつもりでそんなことを言ったのかはらちんねん！\n"
            "日本人のアシスタントだぞ。日本のことをよく知っているはずだ。 </s>"
            "<s>[INST]日本の一番高い山は？  [/INST]"
            "  日本の一番高い山は、富士山（3,776m)です。 </s>"
            "<s>[INST]2000メートル以下の一番高い山は？  [/INST]"
            " 日本では、富士山（3,776m)が2000メートルの山を超えているので、日本の2000メートルの一番高い山は富士山です。しかし、他の国では、2000メートルの山が別の名称で知られていることもあります。例えば、日本の白山は1986メートルの山ですが、日本人でなくても登ることができます。 </s>"
            "<s>[INST]ハイキングするための一番いい季節は？  [/INST]"
            "  日本では、夏がおすすめです。 8月は、夏休みの時期なので、多くの人が旅行に出かけます。 また、夏は、気温が上がるため、雪山が崩壊しやすい気候です。 そのため、夏の山の旅はおすすめです。</s>"
            "<s>[INST]夏の湿気が高すぎて辛いです。他の季節のおすすめはありますか。？  [/INST]"
            "  一般的には、春 (4~5月)  と秋 (9~10月) がお勧めです。\n"
            "春は、山の旅のメッポルが良く、 季節の花粉を眺める事ができます。 秋は、空間の青空を眺める事ができます。\n"
            "但し、この二つのシーズンは、山の旅としては、人が多く、 取り易い旅程では、ございません。\n"
            "お互いに、数日間の休暇をあてにすることが、 旅の満足を高めることのであります。</s>"
            "<s>[INST]今年の冬休みの準備をしています。日本は広いので、南日本の観光地域のおすすめありますでしょうか。 [/INST]"
            ;
    }

    // calm2-7b-chat
    {
        // https://huggingface.co/cyberagent/calm2-7b-chat
        model_filename = "/tmp/test/ggml-model-f32-q5_k_m.gguf";
        // prompt = "USER: AIによって私達の暮らしはどのように変わりますか？\n"
        //     "ASSISTANT: ";
        prompt = "USER: こんにちは。\n"
            "ASSISTANT: こんにちは！何かお手伝いできることがあれば遠慮なくおっしゃってくださいね。\n"
            "USER: 日本の一番高い山は？\n"
            "ASSISTANT: ";
    }

    // RakutenAI-7B-chat-q8_0
    {
        model_filename = "/tmp/test/RakutenAI-7B-chat-q8_0.gguf";
        prompt = "USER: こんにちは。\nASSISTANT: ";

        prompt = "";
        // prompt += "USER: こんにちは。\n";
        // prompt += "ASSISTANT: こんにちは。どういったご用件でしょうか？\n";

        // test prompt to force the model to generate multibyte utf8 characters
        prompt += "USER: 顔文字を書いてください。例えば「(ง ื▿ ื)ว」\n";
        prompt += "ASSISTANT: ";

        std::cout << "\n\nprompt:\n";
        std::cout << prompt;
        std::cout << "\n-----------------\n";
    }

    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--model", strlen("--model"))) {
            if (i + 1 < argc) {
                model_filename = std::string(argv[i + 1]);
                i++;
            }
        }
        if (!strncmp(argv[i], "--prompt", strlen("--prompt"))) {
            if (i + 1 < argc) {
                prompt = std::string(argv[i + 1]);
                i++;
            }
        }
    }

    std::cout << "\n\nprompt:\n";
    std::cout << prompt;
    std::cout << "\n-----------------\n";


    LOG("%s: llama backend init\n", __func__);
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    llama_model_params model_params;
    model_params.n_gpu_layers = 0;
    model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
    model_params.main_gpu = 0;
    model_params.vocab_only = false;
    model_params.use_mmap = true;
    model_params.check_tensors = true;

    llama_model* model = llama_load_model_from_file(model_filename.c_str(), model_params);
    if (model == nullptr) {
        std::cout << "error: llama_load_model_from_file" << std::endl;
        return 1;
    }

    struct simple_llama* sllm = simple_llama_new();
    struct simple_llama_inference_state* inference_state = nullptr;

    if (sllm == NULL) {
        std::cout << "error: simple_llama_new" << std::endl;
        return 1;
    }

    std::string acc = "";

    auto status = simple_llama_init_model(sllm, model);
    if (status != SIMPLE_LLAMA_STATUS_SUCCESS) {
        std::cout << "error: simple_llama_init_model: " << status << std::endl;
        goto cleanup;
    }

    inference_state = simple_llama_inference_state_new();
    if (inference_state == NULL) {
        std::cout << "error: simple_llama_inference_state_new" << std::endl;
        goto cleanup;
    }

    if (true) {
        std::vector<simple_llama_chat_message_t> messages;
        simple_llama_chat_message_t cm;
        cm.role = SIMPLE_LLAMA_CHAT_MESSAGE_ROLE_USER;
        cm.content = "顔文字を書いてください。例えば「(°〆°#)」か「(`益′)ว」";
        // cm.content = "最もシンプルなアスキーアートを書いてください";
        messages.push_back(cm);

        // cm.role = SIMPLE_LLAMA_CHAT_MESSAGE_ROLE_ASSISTANT;
        // // cm.content = "（・∀・）つ";
        // cm.content = "どういうテーマのアートですか？";
        // messages.push_back(cm);

        // cm.role = SIMPLE_LLAMA_CHAT_MESSAGE_ROLE_USER;
        // // cm.content = "ascii artで牛を書いてください";
        // cm.content = "人はテーマです。";
        // messages.push_back(cm);

        status = simple_llama_set_prompt_from_messages(
            sllm,
            inference_state,
            messages.data(),
            messages.size(),
            SIMPLE_LLAMA_CHAT_MESSAGE_FORMAT_RAKUTEN_AI_CHAT
        );
    } else {
        status = simple_llama_set_prompt(sllm, inference_state, prompt.c_str());
    }


    if (status != SIMPLE_LLAMA_STATUS_SUCCESS) {
        std::cout << "error: simple_llama_set_prompt: " << status << std::endl;
        goto cleanup;
    }

    simple_llama_inference_status_t inference_status;

    std::cout << "size of char: " << sizeof(char) << std::endl;
    std::cout << "input prompt token count " << simple_llama_inference_state_get_input_prompt_token_count(sllm, inference_state) << std::endl;
    do {
        llama_token token;
        inference_status = simple_llama_inference_state_get_next_token(sllm, inference_state, &token);

        if (inference_status != SIMPLE_LLAMA_INFERENCE_STATUS_SUCCESS) {
            break;
        }

        // Code copied/pasted from common.cpp
        std::vector<char> result(8, 0);
        int n_tokens = simple_llama_token_to_piece(sllm, token, result.data(), result.size());
        if (n_tokens < 0) {
            result.resize(-n_tokens);
            n_tokens = simple_llama_token_to_piece(sllm, token, result.data(), result.size());
            if (n_tokens < 0) {
                // This should not happen
                std::cerr << "simple_llama_token_to_piece returned an error." << std::endl;
                return 1;
            }
        } else {
            result.resize(n_tokens);
        }

        // Note: here result might hold a truncated multibyte character.
        // We can check the length of the character using utf8_len.
        // If the result is greater than n_tokens we should wait before yielding
        // the token.
        std::string tmp = std::string(result.data(), result.size());
        // if (std::string(result.data(), result.size()) == "\n") {
        //     // for AI Rakuten
        //     inference_status = SIMPLE_LLAMA_INFERENCE_STATUS_DONE;
        // } else 
        {
            std::cout << "generated token (#" << n_tokens << ")'" << std::string(result.data(), result.size()) << "'" << std::endl;
            std::cout << "utf8_len() => " << utf8_len(result[0]) << std::endl;
            for (size_t i = 0; i < result.size(); i++) {
                printf("%02X ", (uint8_t)(result[i]));
            }
            printf("\n");
            acc += std::string(result.data(), result.size());
            std::cout << "acc: " << acc << "\n\n" << std::endl;
        }
    } while(inference_status == SIMPLE_LLAMA_INFERENCE_STATUS_SUCCESS);

    std::cout << "\n\n---------------- prompt\n" << prompt << "\n----------------\n" << std::endl;
    std::cout << "\n\n---------------- answer\n" << acc << "\n----------------\n" << std::endl;

cleanup:
    if (inference_state != nullptr) {
        simple_llama_inference_state_free(inference_state);
    }

    simple_llama_free(sllm);

    llama_backend_free();

#ifndef LOG_DISABLE_LOGS
    LOG_TEE("Log end\n");
#endif // LOG_DISABLE_LOGS

    return 0;
}
