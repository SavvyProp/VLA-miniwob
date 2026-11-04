# Third-party code and formats

- **MiniWoB++**, Farama Foundation and contributors, MIT License. Pinned at
  `33c3b4ddef8c6eb67c57a29663d844b1eda7e614`. Supplies Selenium browser lifecycle,
  clicks, screenshots, and core HTML/JavaScript assets. License retained upstream:
  https://github.com/Farama-Foundation/miniwob-plusplus/blob/33c3b4ddef8c6eb67c57a29663d844b1eda7e614/LICENSE
- **SeeClick**, its authors, Apache License 2.0 for the repository code. Training
  and evaluation load its pretrained checkpoint and pinned original Qwen-VL custom
  implementation from Hugging Face. No model implementation or weights are vendored.
  The export follows its conversation schema and MiniWoB click convention.
  Loader: https://github.com/njucckevin/SeeClick/blob/main/finetune/finetune.py
  Preprocessing: https://github.com/njucckevin/SeeClick/blob/main/agent_tasks/miniwob_process.py
- **Qwen-VL**, Alibaba Cloud and contributors. Pinned tokenizer/model custom code
  and font are loaded as runtime dependencies. Consult the model repository's terms:
  https://huggingface.co/Qwen/Qwen-VL-Chat
- **Transformers, PEFT, Accelerate**, Hugging Face and contributors, Apache License
  2.0. Used for model loading, LoRA, training, and checkpoints. **PyTorch/torchvision**
  and **bitsandbytes** are optional training-runtime dependencies with their upstream
  licenses retained. All package versions are frozen in the integration lockfile.
- **OpenAI Python SDK**, Apache License 2.0. Used for Responses requests with
  structured actions: https://github.com/openai/openai-python

The scriptWoB grammar, interpreter, pages, and collection/export implementation are
custom code. This is not an unchanged MiniWoB++/CompWoB reproduction. No pretrained
model weights are redistributed; consult upstream model/data terms when using them.
