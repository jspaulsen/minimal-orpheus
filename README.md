# minimal-orpheus

## Installation
```bash
export CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CCACHE=OFF -DCMAKE_CXX_STANDARD=17 -DCMAKE_CUDA_ARCHITECTURES=86;89;120 CMAKE_PARALLEL_LEVEL=16"
export FORCE_CMAKE=1

uv sync
```

## Usage
`main.py` has a simple example of how to use Orpheus. The TL;DR is Orpheus returns a generator of bytes.

The format is 16 bit signed PCM, 24khz, mono audio. You can use `wave` to save it to a file, instead of to BytesIO.

## Troubleshooting

If it fails randomly (can't load model, etc.,) reinstall llama-cpp-python explicitly:
```bash
uv add llama-cpp-python --no-cache --reinstall
```

## Other notes
`Llama` is a light wrapper over the `llama-cpp-python` to try and abstract a little bit of the complexity away. Dunno if it's all that useful.
