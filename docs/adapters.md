# Framework adapters

Already using LangChain or LlamaIndex? Wrap your existing retriever/chain instead of rewriting it as plain functions.

## LangChain

```bash
pip install ragmark[langchain]
```

```python
from rag_score.adapters.langchain_adapter import (
    LangChainRetrieverAdapter,
    LangChainGeneratorAdapter,
)

retriever = LangChainRetrieverAdapter(my_vectorstore.as_retriever())
generator = LangChainGeneratorAdapter(my_lcel_chain)
```

- [`LangChainRetrieverAdapter`][rag_score.adapters.langchain_adapter.LangChainRetrieverAdapter] wraps any `langchain_core.retrievers.BaseRetriever`. Document `doc_id` is read from metadata (`doc_id`, then `id`, then `source`, whichever is present first).
- [`LangChainGeneratorAdapter`][rag_score.adapters.langchain_adapter.LangChainGeneratorAdapter] wraps any `Runnable` - an LLM, a chat model, or a full LCEL chain. It passes `{"question": ..., "context": ...}` by default; pass a custom `input_formatter` if your chain expects a different shape. Output is unwrapped automatically whether your chain returns a plain string, an `AIMessage`, or a dict with an `answer`/`output`/`text`/`result` key.

See `examples/langchain_example.py` for a full runnable version.

## LlamaIndex

```bash
pip install ragmark[llamaindex]
```

```python
from rag_score.adapters.llamaindex_adapter import (
    LlamaIndexRetrieverAdapter,
    LlamaIndexGeneratorAdapter,
)

retriever = LlamaIndexRetrieverAdapter(my_index.as_retriever())
generator = LlamaIndexGeneratorAdapter(my_index.as_query_engine())  # or a bare LLM
```

- [`LlamaIndexRetrieverAdapter`][rag_score.adapters.llamaindex_adapter.LlamaIndexRetrieverAdapter] wraps any `llama_index.core.base.base_retriever.BaseRetriever`.
- [`LlamaIndexGeneratorAdapter`][rag_score.adapters.llamaindex_adapter.LlamaIndexGeneratorAdapter] auto-detects whether it's wrapping a full query engine (`.aquery`) or a bare LLM (`.acomplete`). Query engines retrieve internally, so when pairing this with `LlamaIndexRetrieverAdapter` in the same run, retrieval happens twice - each is scored/timed independently, which is expected.

See `examples/llamaindex_example.py` for a full runnable version.

## Writing your own adapter

No framework you use? See [`RetrieverAdapter`][rag_score.adapters.base.RetrieverAdapter] and [`GeneratorAdapter`][rag_score.adapters.base.GeneratorAdapter] - two-method interfaces. `CallableRetrieverAdapter`/`CallableGeneratorAdapter` wrap plain functions (sync or async) with zero subclassing, which covers the vast majority of custom pipelines - see `examples/raw_example.py`.
