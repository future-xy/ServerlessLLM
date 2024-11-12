# ServerlessLLM Docker Compose Quickstart

```bash
docker compose up -d --build
```

```bash
sllm-cli deploy --config config-opt-2.7b.json
sllm-cli deploy --config config-opt-1.3b.json
```

```bash
sllm-cli generate input-opt-2.7b.json &
sleep 3
sllm-cli generate input-opt-1.3b.json
```