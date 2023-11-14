# OpenAI-Compatible RESTful APIs & SDK

FastChat provides OpenAI-compatible APIs for its supported models, so you can use FastChat as a local drop-in replacement for OpenAI APIs.
The FastChat server is compatible with both [openai-python](https://github.com/openai/openai-python) library and cURL commands.

The following OpenAI APIs are supported:
- Chat Completions. (Reference: https://platform.openai.com/docs/api-reference/chat)
- Completions. (Reference: https://platform.openai.com/docs/api-reference/completions)
- Embeddings. (Reference: https://platform.openai.com/docs/api-reference/embeddings)

## RESTful API Server
First, launch the controller

```bash
python3 -m fastchat.serve.controller
```

Then, launch the model worker(s)

```bash
python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.3
```

Finally, launch the RESTful API server

```bash
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```

Now, let us test the API server.

### OpenAI Official SDK
The goal of `openai_api_server.py` is to implement a fully OpenAI-compatible API server, so the models can be used directly with [openai-python](https://github.com/openai/openai-python) library.

First, install openai-python:
```bash
pip install --upgrade openai
```

Then, interact with model vicuna:
```python
import openai
# to get proper authentication, make sure to use a valid key that's listed in
# the --api-keys flag. if no flag value is provided, the `api_key` will be ignored.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

model = "vicuna-7b-v1.3"
prompt = "Once upon a time"

# create a completion
completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=64)
# print the completion
print(prompt + completion.choices[0].text)

# create a chat completion
completion = openai.ChatCompletion.create(
  model=model,
  messages=[{"role": "user", "content": "Hello! What is your name?"}]
)
# print the completion
print(completion.choices[0].message.content)
```

Streaming is also supported. See [test_openai_api.py](../tests/test_openai_api.py).

### cURL
cURL is another good tool for observing the output of the api.

List Models:
```bash
curl http://localhost:8000/v1/models
```

Chat Completions:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vicuna-7b-v1.3",
    "messages": [{"role": "user", "content": "Hello! What is your name?"}]
  }'
```

Text Completions:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vicuna-7b-v1.3",
    "prompt": "Once upon a time",
    "max_tokens": 41,
    "temperature": 0.5
  }'
```

Embeddings:
```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vicuna-7b-v1.3",
    "input": "Hello world!"
  }'
```

### Running multiple 

If you want to run multiple models on the same machine and in the same process,
you can replace the `model_worker` step above with a multi model variant:

```bash
python3 -m fastchat.serve.multi_model_worker \
    --model-path lmsys/vicuna-7b-v1.3 \
    --model-names vicuna-7b-v1.3 \
    --model-path lmsys/longchat-7b-16k \
    --model-names longchat-7b-16k
```

This loads both models into the same accelerator and in the same process.  This
works best when using a Peft model that triggers the `PeftModelAdapter`.

TODO: Base model weight optimization will be fixed once [this
Peft](https://github.com/huggingface/peft/issues/430) issue is resolved.

## LangChain Support
This OpenAI-compatible API server supports LangChain. See [LangChain Integration](langchain_integration.md) for details.

## Adjusting Environment Variables

### Timeout
By default, a timeout error will occur if a model worker does not response within 100 seconds. If your model/hardware is slower, you can change this timeout through an environment variable: 

```bash
export FASTCHAT_WORKER_API_TIMEOUT=<larger timeout in seconds>
```

### Batch size
If you meet the following OOM error while creating embeddings. You can use a smaller batch size by setting

```bash
export FASTCHAT_WORKER_API_EMBEDDING_BATCH_SIZE=1
```

## Todos
Some features to be implemented:

- [ ] Support more parameters like `logprobs`, `logit_bias`, `user`, `presence_penalty` and `frequency_penalty`
- [ ] Model details (permissions, owner and create time)
- [ ] Edits API
- [ ] Rate Limitation Settings
