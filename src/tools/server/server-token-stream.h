#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

// Text emission can lag sampling while UTF-8 is incomplete. Return every
// sampled token since the previous emission, including at stream termination.
template <typename Token>
std::vector<Token> server_stream_pending_tokens(const std::vector<Token> & generated, size_t & sent) {
    if (sent > generated.size()) {
        throw std::logic_error("stream token cursor exceeds generated token count");
    }
    std::vector<Token> pending(generated.begin() + sent, generated.end());
    sent = generated.size();
    return pending;
}
