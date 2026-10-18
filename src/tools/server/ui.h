// MONSTER: WebUI removed; empty asset registry stub (interface preserved for server-http.cpp)
#pragma once
#include <string>
#include <vector>
struct llama_ui_asset {
    std::string name;
    std::string mime_type;
    std::string content;
};
inline const std::vector<llama_ui_asset> & llama_ui_get_assets() {
    static const std::vector<llama_ui_asset> empty;
    return empty;
}
