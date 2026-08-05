// =============================================================================
// http_server.h — 极简 HTTP/1.1 服务器 (header-only, Winsock2, 零外部依赖)
// =============================================================================
// 设计取舍 (为 OpenAI 兼容本地服务定制):
//   - 单线程顺序处理: accept 一个连接 → 完整读请求 → handler → 回响应 → close。
//     引擎 (SNN + LLM) 天然单会话, 顺序处理避免并发互斥; 客户端请求同步等待,
//     排队可接受 (一轮 ≈ 2-6s)。
//   - 只支持 Content-Length 定长请求体 (OpenAI 客户端均如此, 不支持 chunked)。
//   - 响应固定 Connection: close, 每请求新连接, 免 keep-alive 状态管理。
//   - 监听 127.0.0.1 (仅本机), 不对外暴露。
// 使用:
//   HttpServer srv;
//   srv.run(port, [](const HttpRequest& req, HttpResponse& resp) -> bool {
//       resp.status = 200; resp.body = "..."; return true;  // false = 停止服务
//   });
// =============================================================================

#ifndef HETERO_BRAIN_HTTP_SERVER_H
#define HETERO_BRAIN_HTTP_SERVER_H

#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#error "http_server.h: 当前仅支持 Windows (Winsock2)"
#endif

namespace hb {
namespace net {

struct HttpRequest {
    std::string method;   // "GET" / "POST" / ...
    std::string path;     // 含 query 的原始路径 (如 /v1/models?x=1)
    std::string body;
    std::vector<std::pair<std::string, std::string>> headers;

    // 大小写不敏感取头部 (未命中返回空串)
    std::string header(const std::string& name) const {
        for (const auto& kv : headers) {
            if (kv.first.size() == name.size()) {
                bool eq = true;
                for (size_t i = 0; i < name.size(); ++i) {
                    const char a = kv.first[i], b = name[i];
                    if ((a | 0x20) != (b | 0x20)) { eq = false; break; }
                }
                if (eq) {
                    return kv.second;
                }
            }
        }
        return std::string();
    }
};

struct HttpResponse {
    int status = 200;
    std::string body;
    // 显式 charset: 若缺省, 部分客户端 (如 .NET HttpWebRequest) 会按 Latin1
    // 解码响应体 → UTF-8 中文变乱码 (2026-08-05 实测)
    std::string content_type = "application/json; charset=utf-8";
    bool send_cors = true;   // 兼容 Electron 壳子
};

class HttpServer {
public:
    // 返回值: 0 正常结束 (handler 返回 false 或 Ctrl+C 中断); -1 初始化失败; -2 绑定失败
    int run(int port, const std::function<bool(const HttpRequest&, HttpResponse&)>& handler) {
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
            std::fprintf(stderr, "[http] WSAStartup failed\n");
            return -1;
        }
        SOCKET listen_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (listen_fd == INVALID_SOCKET) {
            std::fprintf(stderr, "[http] socket failed: %d\n", WSAGetLastError());
            WSACleanup();
            return -1;
        }
        int yes = 1;
        setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(yes));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);   // 仅本机
        addr.sin_port = htons((u_short)port);
        if (bind(listen_fd, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
            std::fprintf(stderr, "[http] bind 127.0.0.1:%d failed: %d (端口被占用?)\n",
                         port, WSAGetLastError());
            closesocket(listen_fd);
            WSACleanup();
            return -2;
        }
        if (listen(listen_fd, 8) == SOCKET_ERROR) {
            std::fprintf(stderr, "[http] listen failed: %d\n", WSAGetLastError());
            closesocket(listen_fd);
            WSACleanup();
            return -1;
        }
        std::printf("[http] OpenAI 兼容服务已启动: http://127.0.0.1:%d/v1\n", port);
        std::printf("[http] 端点: GET /v1/models | POST /v1/chat/completions\n");
        std::fflush(stdout);

        int rc = 0;
        for (;;) {
            SOCKET fd = accept(listen_fd, nullptr, nullptr);
            if (fd == INVALID_SOCKET) {
                std::fprintf(stderr, "[http] accept failed: %d\n", WSAGetLastError());
                rc = -1;
                break;
            }
            HttpRequest req;
            HttpResponse resp;
            const bool keep = handle_connection(fd, req, resp);
            if (keep) {
                const bool cont = handler ? handler(req, resp) : true;
                if (!cont) {
                    closesocket(fd);
                    rc = 0;
                    break;
                }
            }
            send_response(fd, resp);
            closesocket(fd);
        }
        closesocket(listen_fd);
        WSACleanup();
        return rc;
    }

private:
    // 读完整请求 (头 + Content-Length 定长体)。返回 true 表示有有效请求。
    bool handle_connection(SOCKET fd, HttpRequest& req, HttpResponse& /*resp*/) {
        std::string buf;
        char tmp[8192];
        // ---- 读请求头 (到 \r\n\r\n) ----
        while (buf.find("\r\n\r\n") == std::string::npos) {
            const int n = recv(fd, tmp, sizeof(tmp), 0);
            if (n <= 0) {
                return false;   // 连接关闭/错误
            }
            buf.append(tmp, (size_t)n);
            if (buf.size() > (1u << 20)) {
                return false;   // 头部过大, 丢弃
            }
        }
        const size_t hdr_end = buf.find("\r\n\r\n");
        const std::string head = buf.substr(0, hdr_end);
        std::string body_rest = buf.substr(hdr_end + 4);

        // ---- 解析请求行 ----
        const size_t sp1 = head.find(' ');
        const size_t sp2 = head.find(' ', sp1 + 1);
        if (sp1 == std::string::npos || sp2 == std::string::npos) {
            return false;
        }
        req.method = head.substr(0, sp1);
        req.path = head.substr(sp1 + 1, sp2 - sp1 - 1);

        // ---- 解析头部 ----
        size_t pos = sp2 + 1;
        while (pos < head.size()) {
            const size_t eol = head.find("\r\n", pos);
            if (eol == std::string::npos) {
                break;
            }
            const std::string line = head.substr(pos, eol - pos);
            const size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string name = line.substr(0, colon);
                std::string val = line.substr(colon + 1);
                while (!val.empty() && (val.front() == ' ' || val.front() == '\t')) {
                    val.erase(val.begin());
                }
                while (!val.empty() && (val.back() == ' ' || val.back() == '\t')) {
                    val.pop_back();
                }
                req.headers.emplace_back(std::move(name), std::move(val));
            }
            pos = eol + 2;
        }

        // ---- 读请求体 (Content-Length) ----
        const std::string cl = req.header("Content-Length");
        size_t need = 0;
        if (!cl.empty()) {
            need = (size_t)std::strtoull(cl.c_str(), nullptr, 10);
            if (need > (1u << 24)) {
                return false;   // 请求体过大 (16MB 上限)
            }
        }
        if (body_rest.size() < need) {
            while (body_rest.size() < need) {
                const int n = recv(fd, tmp, sizeof(tmp), 0);
                if (n <= 0) {
                    return false;
                }
                body_rest.append(tmp, (size_t)n);
            }
        }
        req.body = body_rest.substr(0, need);
        return true;
    }

    void send_response(SOCKET fd, const HttpResponse& resp) {
        std::string head;
        char st[64];
        std::snprintf(st, sizeof(st), "HTTP/1.1 %d ", resp.status);
        head += st;
        switch (resp.status) {
            case 200: head += "OK"; break;
            case 204: head += "No Content"; break;
            case 400: head += "Bad Request"; break;
            case 401: head += "Unauthorized"; break;
            case 404: head += "Not Found"; break;
            case 500: head += "Internal Server Error"; break;
            default:  head += "Unknown"; break;
        }
        head += "\r\n";
        head += "Content-Type: " + resp.content_type + "\r\n";
        if (resp.status != 204) {
            char lb[32];
            std::snprintf(lb, sizeof(lb), "%zu", resp.body.size());
            head += std::string("Content-Length: ") + lb + "\r\n";
        }
        if (resp.send_cors) {
            head += "Access-Control-Allow-Origin: *\r\n";
            head += "Access-Control-Allow-Headers: authorization, content-type\r\n";
            head += "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
        }
        head += "Connection: close\r\n\r\n";
        send(fd, head.c_str(), (int)head.size(), 0);
        if (resp.status != 204 && !resp.body.empty()) {
            send(fd, resp.body.c_str(), (int)resp.body.size(), 0);
        }
    }
};

}  // namespace net
}  // namespace hb

#endif  // HETERO_BRAIN_HTTP_SERVER_H
