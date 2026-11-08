// qwen_parse_ref: llama.cpp's own chat-output parser (common/chat*, the Qwen3-Coder / Qwen3.5 PEG parser picked by the
// template) driven from JSONL, so the K3 server's Python parser (python/caminus/serve/toolparse.py) can be gated
// against it (spec-server.md T0, step S2). CPU only, no model: the template comes from a file.
//
// stdin, one JSON object per line:
//   {"tools": [OpenAI tools], "enable_thinking": true, "tool_choice": "auto"|"required"|"none",
//    "parallel": true, "text": "<model output after the generation prompt>", "partial_at": [byte offsets]}
// stdout, one JSON object per line:
//   {"gen_prompt": "...", "final": MSG|null, "error": "..."|null, "partials": [MSG|null|{"error":...}]}
//   MSG = {"reasoning_content": str, "content": str, "tool_calls": [{"name": str, "arguments": str}]}
// The final parse is is_partial=false (llama-server's finish()); each partial_at offset is parsed with is_partial=true
// on text[:offset] (llama-server's feed()).
#include "chat.h"
#include "json.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

static common_json msg_to_json(const common_chat_msg & m) {
    common_json calls = common_json::array();
    for (const auto & tc : m.tool_calls) {
        common_json c = common_json::object();
        c["name"]      = tc.name;
        c["arguments"] = tc.arguments;
        calls.push_back(c);
    }
    common_json o = common_json::object();
    o["reasoning_content"] = m.reasoning_content;
    o["content"]           = m.content;
    o["tool_calls"]        = calls;
    return o;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s template.jinja < cases.jsonl\n", argv[0]);
        return 2;
    }
    std::ifstream f(argv[1]);
    std::stringstream ss;
    ss << f.rdbuf();
    auto tmpls = common_chat_templates_init(nullptr, ss.str(), "", "<|im_end|>");

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        common_json in  = common_json::parse(line);
        common_json out = common_json::object();
        try {
            common_chat_templates_inputs inputs;
            common_chat_msg user;
            user.role    = "user";
            user.content = "hi";
            inputs.messages.push_back(user);
            inputs.use_jinja        = true;
            inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;  // llama-server's default
            inputs.enable_thinking  = in.contains("enable_thinking") ? in.at("enable_thinking").get<bool>() : true;
            inputs.chat_template_kwargs["enable_thinking"] = inputs.enable_thinking ? "true" : "false";
            inputs.parallel_tool_calls = in.contains("parallel") ? in.at("parallel").get<bool>() : true;
            if (in.contains("tools") && !in.at("tools").empty()) {
                inputs.tools = common_chat_tools_parse_oaicompat(in.at("tools"));
            }
            if (in.contains("tool_choice")) {
                inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(in.at("tool_choice").get<std::string>());
            }
            auto params = common_chat_templates_apply(tmpls.get(), inputs);
            common_chat_parser_params pp(params);
            pp.parser = params.parser;
            out["gen_prompt"] = params.generation_prompt;
            out["format"]     = std::string(common_chat_format_name(params.format));

            const std::string text = in.at("text").get<std::string>();
            try {
                out["final"] = msg_to_json(common_chat_parse(common_chat_input(text), false, pp));
                out["error"] = nullptr;
            } catch (const std::exception & e) {
                out["final"] = nullptr;
                out["error"] = std::string(e.what());
            }
            common_json partials = common_json::array();
            if (in.contains("partial_at")) {
                const auto & pa = in.at("partial_at");
                for (size_t i = 0; i < pa.size(); i++) {
                    size_t n = pa.at(i).get<size_t>();
                    try {
                        partials.push_back(msg_to_json(common_chat_parse(common_chat_input(text.substr(0, n)), true, pp)));
                    } catch (const std::exception & e) {
                        common_json err = common_json::object();
                        err["error"] = std::string(e.what());
                        partials.push_back(err);
                    }
                }
            }
            out["partials"] = partials;
        } catch (const std::exception & e) {
            out["setup_error"] = std::string(e.what());
        }
        std::cout << out.dump() << "\n" << std::flush;
    }
    return 0;
}
