#ifndef SIMPLE_LLAMA_H
#define SIMPLE_LLAMA_H

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //

    enum simple_llama_status {
        SIMPLE_LLAMA_STATUS_SUCCESS = 0,
        SIMPLE_LLAMA_STATUS_LOAD_MODEL_FAILURE = 1,
        SIMPLE_LLAMA_STATUS_CONTEXT_CREATION_FAILURE = 2,
        SIMPLE_LLAMA_STATUS_LORA_FAILURE = 3,
        SIMPLE_LLAMA_STATUS_NO_EOS_TOKEN_FOUND = 4,
        SIMPLE_LLAMA_STATUS_EMPTY_PROMPT = 5,
        SIMPLE_LLAMA_STATUS_PROMPT_TOO_LONG = 6,
        SIMPLE_LLAMA_STATUS_CONTEXT_SAMPLING_FAILURE = 7,
    };

    enum simple_llama_inference_status {
        SIMPLE_LLAMA_INFERENCE_STATUS_SUCCESS = 0,
        SIMPLE_LLAMA_INFERENCE_STATUS_DONE = 1, // no more token to produce
        SIMPLE_LLAMA_INFERENCE_STATUS_CONTEXT_FULL = 2,
        SIMPLE_LLAMA_INFERENCE_STATUS_LLAMA_DECODE_FAILURE = 3,
    };

    // wrap gpt_params
    struct simple_llama;
    struct simple_llama_inference_state;

    LLAMA_API struct simple_llama* simple_llama_new();
    LLAMA_API simple_llama_status simple_llama_init_model(struct simple_llama*, const char* model_file);
    LLAMA_API void simple_llama_free(struct simple_llama*);
    LLAMA_API simple_llama_status simple_llama_set_prompt(struct simple_llama*, struct simple_llama_inference_state*, const char* prompt);

    LLAMA_API struct simple_llama_inference_state* simple_llama_inference_state_new();
    LLAMA_API void simple_llama_inference_state_free(struct simple_llama_inference_state*);
    LLAMA_API int simple_llama_inference_state_get_input_prompt_token_count(struct simple_llama*, struct simple_llama_inference_state*);

    // return the next token
    LLAMA_API simple_llama_inference_status simple_llama_inference_state_get_next_token(struct simple_llama*, struct simple_llama_inference_state*, llama_token*);
    // wrapper atop llama_token_to_piece
    LLAMA_API int32_t simple_llama_token_to_piece(struct simple_llama*, llama_token, char * buf, int32_t length);

#ifdef __cplusplus
}
#endif

#endif // SIMPLE_LLAMA_H