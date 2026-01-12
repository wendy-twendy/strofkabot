#!/usr/bin/env python3
"""Compare RAG quality: auto-generated vs LLM-extracted metadata filtering.

Compares filtering effectiveness on the SAME chunks using:
1. Auto-generated metadata (channel_name, participant_names, etc.)
2. LLM-extracted metadata (domain_tags, named_entities, language, etc.)

Usage:
    set -a && source .env && set +a && .venv/bin/python scripts/compare_rag_quality.py
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from strofkabot.rag.embeddings import OpenRouterEmbeddingClient
from strofkabot.rag.vector_store import SearchResult, VectorStore


@dataclass
class TestQuery:
    """Test query comparing auto vs LLM metadata filtering."""

    query: str
    description: str
    # Auto-generated metadata filter
    auto_field: str
    auto_value: str
    # LLM-extracted metadata filter
    llm_field: str
    llm_value: str | list[str] | bool


@dataclass
class ComparisonResult:
    """Results for a single query."""

    query: TestQuery
    # Results from vector search (no filter)
    baseline_count: int
    baseline_results: list[dict]
    # Results filtered by auto-generated metadata
    auto_count: int
    auto_results: list[dict]
    # Results filtered by LLM metadata
    llm_count: int
    llm_results: list[dict]


# Test queries comparing auto vs LLM metadata
TEST_QUERIES = [
    # Politics: channel_name vs domain_tags
    TestQuery(
        query="Albanian elections debate",
        description="Politics: channel vs domain_tags",
        auto_field="channel_name",
        auto_value="politikë",
        llm_field="domain_tags",
        llm_value="politics",
    ),
    TestQuery(
        query="Trump opinions discussion",
        description="Politics: channel vs named_entities",
        auto_field="channel_name",
        auto_value="politikë",
        llm_field="named_entities",
        llm_value="Trump",
    ),
    # Tech: dev channel vs tech domain
    TestQuery(
        query="programming AI chatbots",
        description="Tech: channel vs domain_tags",
        auto_field="channel_name",
        auto_value="dev",
        llm_field="domain_tags",
        llm_value="tech",
    ),
    # Music: radiostacion vs music domain
    TestQuery(
        query="Spotify music sharing",
        description="Music: channel vs domain_tags",
        auto_field="channel_name",
        auto_value="radiostacion",
        llm_field="domain_tags",
        llm_value="music",
    ),
    # Travel: udhëtime vs travel domain
    TestQuery(
        query="Vienna trip recommendations",
        description="Travel: channel vs domain_tags",
        auto_field="channel_name",
        auto_value="udhëtime",
        llm_field="domain_tags",
        llm_value="travel",
    ),
    # Gaming: no dedicated channel vs gaming domain
    TestQuery(
        query="Europa Universalis strategy game",
        description="Gaming: kanapeja vs domain_tags",
        auto_field="channel_name",
        auto_value="kanapeja",
        llm_field="domain_tags",
        llm_value="gaming",
    ),
    # Entity search: participant vs named_entities
    TestQuery(
        query="Elon Musk Tesla discussions",
        description="Entity: participant vs named_entities",
        auto_field="participant_names",
        auto_value="",  # Can't filter by entity in auto metadata
        llm_field="named_entities",
        llm_value=["Elon Musk", "Musk", "Tesla"],
    ),
    TestQuery(
        query="Tirana city life experiences",
        description="Entity: no auto filter vs named_entities",
        auto_field="channel_name",
        auto_value="",  # No specific channel for Tirana
        llm_field="named_entities",
        llm_value="Tirana",
    ),
    # Language filtering (LLM only - auto has no language field)
    TestQuery(
        query="Çfarë mendoni për qeverinë?",
        description="Language: no auto filter vs language=sq",
        auto_field="channel_name",
        auto_value="",
        llm_field="language",
        llm_value="sq",
    ),
    # Question detection (LLM only)
    TestQuery(
        query="unanswered questions help needed",
        description="Questions: no auto filter vs is_question",
        auto_field="channel_name",
        auto_value="",
        llm_field="is_question",
        llm_value=True,
    ),
    # Sentiment (LLM only)
    TestQuery(
        query="complaints problems frustrating",
        description="Sentiment: no auto filter vs negative",
        auto_field="channel_name",
        auto_value="",
        llm_field="sentiment",
        llm_value="negative",
    ),
    # Humor/banter detection
    TestQuery(
        query="jokes memes funny banter",
        description="Humor: kanapeja vs humor_level",
        auto_field="channel_name",
        auto_value="kanapeja",
        llm_field="humor_level",
        llm_value=["mild", "heavy"],
    ),
]


def load_metadata_lookup(path: str) -> dict[tuple[str, str], dict]:
    """Load LLM extractions indexed by (channel, start_time)."""
    with open(path) as f:
        extractions = json.load(f)

    pattern = r"\[Channel: ([^|]+)\| (\d{4}-\d{2}-\d{2} \d{2}:\d{2})"
    lookup = {}
    for ext in extractions:
        text = ext.get("chunk_text", "")
        match = re.search(pattern, text)
        if match:
            key = (match.group(1).strip(), match.group(2))
            lookup[key] = ext.get("metadata", {})
    return lookup


def get_chunk_key(doc: str) -> tuple[str, str] | None:
    """Extract (channel, start_time) key from document."""
    pattern = r"\[Channel: ([^|]+)\| (\d{4}-\d{2}-\d{2} \d{2}:\d{2})"
    match = re.search(pattern, doc)
    if match:
        return (match.group(1).strip(), match.group(2))
    return None


def matches_auto_filter(result: SearchResult, field: str, value: str) -> bool:
    """Check if result matches auto-generated metadata filter."""
    if not value:  # Empty value means no auto filter available
        return True

    actual = result.metadata.get(field, "")
    if isinstance(actual, str):
        return value.lower() in actual.lower()
    return False


def matches_llm_filter(
    llm_metadata: dict | None, field: str, value: str | list[str] | bool
) -> bool:
    """Check if LLM metadata matches filter."""
    if llm_metadata is None:
        return False

    actual = llm_metadata.get(field)
    if actual is None:
        return False

    # Boolean match
    if isinstance(value, bool):
        return actual == value

    # List match (any value in list matches)
    if isinstance(value, list):
        if isinstance(actual, list):
            return any(v.lower() in [a.lower() for a in actual] for v in value)
        return str(actual).lower() in [v.lower() for v in value]

    # String match
    if isinstance(actual, list):
        return any(value.lower() in a.lower() for a in actual)
    return value.lower() in str(actual).lower()


async def run_comparison(
    store: VectorStore,
    embeddings: OpenRouterEmbeddingClient,
    metadata_lookup: dict[tuple[str, str], dict],
    k: int = 20,
) -> list[ComparisonResult]:
    """Run comparison for all test queries."""
    results = []

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] {query.description}")
        print(f"    Query: {query.query}")

        # Get query embedding and search
        query_embedding = await embeddings.embed_query(query.query)
        search_results = store.search(query_embedding, k=k)

        baseline_results = []
        auto_filtered = []
        llm_filtered = []

        for result in search_results:
            chunk_key = get_chunk_key(result.document)
            llm_meta = metadata_lookup.get(chunk_key) if chunk_key else None

            result_info = {
                "chunk_key": chunk_key,
                "channel": result.metadata.get("channel_name", ""),
                "similarity": result.similarity,
                "has_llm_meta": llm_meta is not None,
                "preview": result.document[:100].replace("\n", " "),
            }

            # Only consider chunks that have LLM metadata (fair comparison)
            if llm_meta is None:
                continue

            baseline_results.append(result_info)

            # Check auto filter
            if matches_auto_filter(result, query.auto_field, query.auto_value):
                auto_filtered.append(
                    {
                        **result_info,
                        "auto_match": f"{query.auto_field}={query.auto_value}",
                    }
                )

            # Check LLM filter
            if matches_llm_filter(llm_meta, query.llm_field, query.llm_value):
                llm_filtered.append(
                    {
                        **result_info,
                        "llm_match": f"{query.llm_field}={query.llm_value}",
                        "llm_value": llm_meta.get(query.llm_field),
                    }
                )

        print(f"    Chunks with LLM metadata: {len(baseline_results)}/{k}")
        print(
            f"    Auto filter ({query.auto_field}={query.auto_value or 'N/A'}): {len(auto_filtered)}"
        )
        print(f"    LLM filter ({query.llm_field}): {len(llm_filtered)}")

        # Show advantage
        if query.auto_value:
            diff = len(llm_filtered) - len(auto_filtered)
            if diff > 0:
                print(f"    → LLM advantage: +{diff} results")
            elif diff < 0:
                print(f"    → Auto advantage: +{-diff} results")
            else:
                print("    → Equal results")
        else:
            print(f"    → LLM-only capability: {len(llm_filtered)} results (no auto equivalent)")

        results.append(
            ComparisonResult(
                query=query,
                baseline_count=len(baseline_results),
                baseline_results=baseline_results,
                auto_count=len(auto_filtered),
                auto_results=auto_filtered,
                llm_count=len(llm_filtered),
                llm_results=llm_filtered,
            )
        )

    return results


def print_summary(results: list[ComparisonResult]) -> None:
    """Print comparison summary."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY: Auto-Generated vs LLM Metadata")
    print("=" * 70)

    # Group by category
    comparable = []  # Has both auto and LLM filter
    llm_only = []  # Only LLM can filter

    for r in results:
        if r.query.auto_value:
            comparable.append(r)
        else:
            llm_only.append(r)

    print("\n### Comparable Filters (both auto and LLM can filter)")
    print("-" * 60)
    total_auto = 0
    total_llm = 0
    for r in comparable:
        auto = r.auto_count
        llm = r.llm_count
        total_auto += auto
        total_llm += llm
        winner = "LLM" if llm > auto else ("AUTO" if auto > llm else "TIE")
        print(f"  {r.query.description}")
        print(
            f"    Auto ({r.query.auto_field}): {auto} | LLM ({r.query.llm_field}): {llm} → {winner}"
        )

    print(f"\n  TOTAL: Auto={total_auto}, LLM={total_llm}")
    if total_llm > total_auto:
        print(
            f"  → LLM metadata provides {total_llm - total_auto} more filtered results ({(total_llm/total_auto - 1)*100:.0f}% improvement)"
        )

    print("\n### LLM-Only Capabilities (no auto equivalent)")
    print("-" * 60)
    for r in llm_only:
        print(f"  {r.query.description}")
        print(f"    LLM ({r.query.llm_field}): {r.llm_count} results")

    total_llm_only = sum(r.llm_count for r in llm_only)
    print(f"\n  TOTAL: {total_llm_only} results only possible with LLM metadata")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. CHANNEL vs DOMAIN_TAGS:
   - Channel filtering is rigid (music only in #radiostacion)
   - Domain tags find topic ANYWHERE (music discussed in any channel)

2. LLM-ONLY CAPABILITIES:
   - named_entities: Find discussions about specific people/places
   - language: Filter by Albanian/English/mixed
   - is_question: Find unanswered questions
   - sentiment: Filter by emotional tone
   - humor_level: Find banter vs serious discussions

3. CROSS-CHANNEL DISCOVERY:
   - LLM metadata finds related content across channels
   - Auto metadata limited to single-channel filtering
""")


async def main():
    print("=" * 70)
    print("RAG Comparison: Auto-Generated vs LLM Metadata Filtering")
    print("=" * 70)

    # Load LLM metadata
    print("\nLoading LLM-extracted metadata...")
    metadata_lookup = load_metadata_lookup("/tmp/metadata_test_results.json")
    print(f"  {len(metadata_lookup)} chunks with LLM metadata")

    # Load vector store
    print("\nLoading vector store...")
    store = VectorStore(Path("/home/endi/strofkabot/data/vector_store"))
    print(f"  {store.count()} total chunks")

    # Initialize embeddings
    print("\nInitializing embeddings client...")
    embeddings = OpenRouterEmbeddingClient()

    # Run comparison
    print("\n" + "=" * 70)
    print("Running comparison tests (k=20, only chunks with LLM metadata)...")
    print("=" * 70)

    results = await run_comparison(store, embeddings, metadata_lookup, k=20)

    # Print summary
    print_summary(results)

    # Save results
    output_dir = Path("/home/endi/strofkabot/_temp_metadata_test/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "comparison_report.json"

    output_data = {
        "summary": {
            "comparable_queries": len([r for r in results if r.query.auto_value]),
            "llm_only_queries": len([r for r in results if not r.query.auto_value]),
        },
        "results": [
            {
                "query": r.query.query,
                "description": r.query.description,
                "auto_filter": f"{r.query.auto_field}={r.query.auto_value}",
                "llm_filter": f"{r.query.llm_field}={r.query.llm_value}",
                "baseline_count": r.baseline_count,
                "auto_count": r.auto_count,
                "llm_count": r.llm_count,
            }
            for r in results
        ],
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
