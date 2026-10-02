#!/usr/bin/env python3
"""A simple HTTP server that mimics an OpenAI-compatible endpoint for judging."""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

class JudgeHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            # We don't parse the request, just return a fixed verdict
            response = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1712345678,
                "model": "local-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"score": 0.8, "reasoning": "This is a fixed response for demonstration."}'
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15
                }
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress log messages
        pass

if __name__ == "__main__":
    server = HTTPServer(('localhost', 8000), JudgeHandler)
    print("Starting judge server on http://localhost:8000")
    server.serve_forever()