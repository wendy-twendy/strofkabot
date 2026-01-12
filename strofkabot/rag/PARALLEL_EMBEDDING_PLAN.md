# Parallel Embedding Implementation

## Status: COMPLETED (2026-01-12)

Full ingestion completed successfully with memory-efficient batching.

## What Was Implemented

### 1. `embeddings.py` - Parallel Processing

Added to `EmbeddingResponse`:
- `failed_indices: list[int]` - tracks which text indices failed

Updated `embed_texts()` with new parameters:
- `max_concurrent: int = 50` - rate limiting via asyncio.Semaphore
- `max_retries: int = 3` - retry attempts per batch
- `retry_delay: float = 1.0` - base delay for exponential backoff (1s, 2s, 4s)

Features:
- Parallel batch processing with semaphore-based rate limiting
- Exponential backoff retry logic
- Order preservation via batch index tracking
- Partial failure tracking with `failed_indices`

### 2. `ingestion.py` - Integration

Updated `generate_embeddings()`:
- Added `max_concurrent` parameter (default 50)
- Passes retry parameters to embedding client
- Raises on any partial failure (ensures chunk/embedding count match)

### 3. `vector_store.py` - Metadata Updates

Added `update_metadata()` method:
- Allows adding LLM metadata after initial ingestion
- Automatically extends LanceDB schema for new fields
- Preserves existing embeddings and metadata

### 4. Tests

Added 6 new tests in `tests/rag/test_embeddings.py`:
- `test_parallel_execution_uses_semaphore`
- `test_retry_on_transient_error`
- `test_partial_failure_returns_successful_embeddings`
- `test_order_preserved_in_parallel`
- `test_failed_indices_tracked_after_all_retries_exhausted`
- `test_new_parameters_have_defaults`

## Performance Results

Verified with real API:

| Test | Time | Result |
|------|------|--------|
| 100 texts (5 batches, 10 concurrent) | 1.39s | 100/100 |
| 200 texts sequential | 4.00s | 200/200 |
| 200 texts parallel (4 concurrent) | 1.35s | 200/200 |

**3x speedup** with just 4 batches. Full run with 50 concurrent expected: **~50x speedup**.

## Full Run Instructions

### Pre-run Checklist

1. Ensure `.env` has `OPENROUTER_API_KEY` set
2. Verify database exists: `data/db.sqlite3`
3. Check available disk space (~500MB for vector store)

### Clear Previous Data

```bash
# Remove existing vector store and BM25 index
rm -rf data/vector_store/

# Verify cleared
ls data/vector_store/ 2>/dev/null || echo "Vector store cleared"
```

### Run Full Ingestion

```bash
# Activate environment and run
set -a && source .env && set +a

# Full run with clear flag (no LLM metadata)
.venv/bin/python scripts/ingest_year.py --clear
```

### Expected Output

Based on current data (2.8M messages):
- **Messages**: ~2,795,527
- **Chunks**: ~110,000 (estimated)
- **Embedding time**: ~20-30 seconds (parallel)
- **Total time**: ~5-10 minutes (including chunking, BM25)

### Monitor Progress

```bash
# Watch progress file
watch -n 5 cat data/vector_store/ingestion_progress.json
```

## Memory Considerations

**WARNING**: Full 2.8M messages loaded into memory at once.

Estimated peak memory usage:
- Messages: ~1.5 GB
- Chunks: ~500 MB
- Embeddings: ~1.3 GB (during generation)
- **Total**: ~3-4 GB RAM required

If memory is a concern, process by year:
```bash
# Process year by year
.venv/bin/python scripts/ingest_year.py --clear --after 2018-01-01 --before 2019-01-01
.venv/bin/python scripts/ingest_year.py --after 2019-01-01 --before 2020-01-01
# ... etc
```

## Post-run Verification

```bash
# Check results
.venv/bin/python3 -c "
from pathlib import Path
from strofkabot.rag.vector_store import VectorStore
store = VectorStore(persist_dir=Path('data/vector_store'))
print(f'Chunks: {store.count():,}')
"

# Test search

.venv/bin/python3 -c "
from pathlib import Path
from strofkabot.rag.pipeline import RAGPipeline
from strofkabot.config import NICKNAMES_FILE, RAG_VECTOR_STORE_DIR
from strofkabot.utils.nickname_loader import load_nicknames
import asyncio

async def test():
    nicknames = load_nicknames(NICKNAMES_FILE)
    pipeline = RAGPipeline(
        vector_store_dir=RAG_VECTOR_STORE_DIR,
        nicknames=nicknames,
    )
    results = await pipeline.search('test query', k=3)
    print(f'Search returned {len(results)} results')

asyncio.run(test())
"
```

## Adding LLM Metadata Later

LLM metadata extraction is skipped for initial ingestion (speed).

To add later:
```python
from strofkabot.rag.vector_store import VectorStore

store = VectorStore(persist_dir=Path('data/vector_store'))

# Update specific chunks with LLM-extracted metadata
store.update_metadata(
    chunk_ids=['chunk-uuid-1', 'chunk-uuid-2'],
    new_metadata=[
        {'topic': 'music', 'sentiment': 'positive'},
        {'topic': 'sports', 'sentiment': 'neutral'},
    ],
)
```

## Troubleshooting

### OOM (Out of Memory)
- Process by year with `--after` and `--before` flags
- Close other applications
- Increase swap space

### API Rate Limits
- Reduce `max_concurrent` in `ingestion.py` (default 50)
- Retry logic handles transient 429 errors automatically

### Partial Failures
- Check logs for specific batch failures
- Ingestion raises on any failure (no partial data)
- Re-run with same parameters to retry

---

## Session 2026-01-12: Full Ingestion & Optimizations

### Final Ingestion Results

| Metric | Value |
|--------|-------|
| Messages processed | 2,796,241 |
| Chunks created | 88,413 |
| Embeddings | 88,413 |
| BM25 indexed | 88,413 |
| LanceDB size | 1.3 GB |
| BM25 index | 376 MB |
| Time periods | 95 (30 days each) |
| Total time | ~6 minutes |

### Memory-Efficient Batching (ingestion.py)

**Problem**: Loading all 2.8M messages caused OOM on 7.5GB server.

**Solution**: Process in 30-day time periods with incremental saving.

New parameters in `IngestionPipeline.run()`:
- `batch_days: int = 30` - days per processing batch
- `resume: bool = True` - resume from previous progress

Flow per batch:
1. Load messages for time period
2. Create chunks
3. Generate embeddings (parallel)
4. Save to vector store immediately
5. Free memory with `gc.collect()`
6. Track progress for resumability

### Chunk Size Limiting (ingestion.py)

**Problem**: Some conversation chunks were 76k chars (~19k tokens), wasting embedding tokens.

**Solution**: Split large chunks by message count.

New parameter:
- `max_chunk_chars: int = 8000` (~2000 tokens max per chunk)

Method `_split_large_chunks()`:
- Uses 70% of max to account for message length variance
- Splits by estimated messages per sub-chunk
- Re-formats each sub-chunk with proper metadata

Results:
- Before: max 76,711 chars, avg batch 257k tokens
- After: max 7,921 chars, avg batch 19k tokens

### BM25 Memory Issue

**Problem**: BM25 index (376MB on disk) expands to ~4GB in RAM when loaded.

**Root cause**: `BM25Index` stores:
- 88k chunk_ids (strings)
- 88k full documents (strings)
- 88k tokenized corpus (list of token lists)
- BM25Okapi object

**Temporary fix**: Disabled BM25 in `cogs/rag.py`:
```python
self._pipeline = RAGPipeline(
    ...
    use_bm25=False,  # Disabled: BM25 index uses ~4GB RAM
)
```

**Future options**:
1. Don't store documents in BM25 (just chunk_ids, fetch from LanceDB)
2. Use streaming/lazy loading
3. Increase server RAM to 16GB+

### Pipeline Changes (pipeline.py)

Added `use_bm25` parameter to `__init__`:
```python
def __init__(self, ..., use_bm25: bool = True):
    self.use_bm25 = use_bm25
```

`semantic_search()` now respects instance default:
```python
async def semantic_search(self, ..., use_bm25: bool | None = None):
    if use_bm25 is None:
        use_bm25 = self.use_bm25
```

### RAG Quality Verification

Tested 12 questions about server history:

| Question | Result |
|----------|--------|
| Server owner | Shark (correct) |
| Admins | Pasa, Takarak, Gogo Soros, Sharku |
| Who is allmight | Found introduction message |
| Why mami kicked | Told Shark to STFU + conspiracy theories |
| Israel/Palestine debaters | Jordi, Igi, Ping, Diti, Beqo, Ardi, Papa |

Vector-only search (no BM25) produces good results.
