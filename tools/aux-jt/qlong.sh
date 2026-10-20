#!/bin/bash
# greedy 300-token story; prints start and end of the reply
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Tell me a short story about a lighthouse keeper.\"}],\"max_tokens\":300,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" | python3 -c "import json,sys; d=json.load(sys.stdin); c=d[\"choices\"][0][\"message\"][\"content\"]; print(repr(c[:100]), \"...\", repr(c[-160:])); print(d[\"timings\"][\"predicted_per_second\"])"
