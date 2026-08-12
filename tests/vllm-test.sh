#!/bin/bash -e

export MODEL="${MODEL:-Qwen/Qwen3-32B}"
export HOST=localhost
export PORT=8000
export ENDPOINT=http://$HOST:$PORT

KEY=sk-xxx

P=" \
The little penguin counted 42 ★ \
The little penguin counted 9 ★ \
The little penguin counted 102 ★ \
The little penguin counted 5 ★ \
Question: How many total counts did the little penguin make, and how many stars per count? \
"

curl --no-buffer --silent --show-error $ENDPOINT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"$P\"}],
  \"temperature\": 0,
  \"stream\": false
  }"
