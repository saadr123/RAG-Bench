"""
RAG-Bench results dashboard.

Run with: streamlit run dashboard/app.py

Reads every results/<config_name>.json file produced by src/runner.py and
gives three views: an overview comparison table, a quality-vs-cost scatter,
and a per-query drill-down for any individual configuration.
"""

from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="RAG-Bench", layout="wide")

RESULTS_DIR = Path("results")


@st.cache_data
def load_all_results() -> dict:
    results = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with open(path) as f:
            results[path.stem] = json.load(f)
    return results


def summarize(config_name: str, data: dict) -> dict:
    queries = data.get("queries", {})
    if not queries:
        return None

    scores = [q["judge_score"] for q in queries.values() if q["judge_score"] is not None]
    n_unscored = len(queries) - len(scores)   # judge output that couldn't be parsed
    judge_costs = [q.get("judge_cost_usd", 0.0) for q in queries.values()]
    hit_rates = [q["retrieval"]["hit_at_k"] for q in queries.values()]
    precisions = [q["retrieval"]["precision_at_k"] for q in queries.values()]
    recalls = [q["retrieval"]["recall_at_k"] for q in queries.values()]
    costs = [q["estimated_cost_usd"] for q in queries.values()]
    latencies = [q["latency_seconds"] for q in queries.values()]

    # config_name convention: "<strategy>_top<k>"
    strategy, top_k_part = config_name.rsplit("_top", 1)

    return {
        "config_name": config_name,
        "chunking_strategy": strategy,
        "top_k": int(top_k_part),
        "n_queries": len(queries),
        "n_unscored": n_unscored,
        "avg_judge_score": sum(scores) / len(scores) if scores else None,
        "hit_rate_at_k": sum(hit_rates) / len(hit_rates),
        "avg_precision_at_k": sum(precisions) / len(precisions),
        "avg_recall_at_k": sum(recalls) / len(recalls),
        "total_cost_usd": sum(costs),
        "judge_cost_usd": sum(judge_costs),
        "avg_cost_per_query_usd": sum(costs) / len(costs),
        "avg_latency_seconds": sum(latencies) / len(latencies),
    }


def main():
    st.title("RAG-Bench: Configuration Sweep Results")

    all_results = load_all_results()
    if not all_results:
        st.warning(
            "No results found in results/. Run `python -m src.runner` first "
            "(add --sample-size 2 for a cheap smoke test)."
        )
        return

    summaries = [summarize(name, data) for name, data in all_results.items()]
    summaries = [s for s in summaries if s is not None]
    df = pd.DataFrame(summaries)

    tab_overview, tab_tradeoff, tab_drilldown = st.tabs(
        ["Overview", "Quality vs Cost", "Drill Down"]
    )

    with tab_overview:
        st.subheader("All configurations")
        st.caption(
            "Costs are MODELED: real token counts priced against a published "
            "rate card, not money actually spent (the sweep runs on a free tier). "
            "`n_unscored` counts queries where the judge returned unparseable "
            "output - a config with several of those isn't fairly comparable."
        )
        total_unscored = int(df["n_unscored"].sum())
        if total_unscored:
            st.warning(
                f"{total_unscored} queries across all configs have no judge score "
                "and are excluded from the quality averages."
            )
        st.dataframe(
            df.sort_values("avg_judge_score", ascending=False).style.format(
                {
                    "avg_judge_score": "{:.2f}",
                    "hit_rate_at_k": "{:.1%}",
                    "avg_precision_at_k": "{:.1%}",
                    "avg_recall_at_k": "{:.1%}",
                    "total_cost_usd": "${:.4f}",
                    "judge_cost_usd": "${:.4f}",
                    "avg_cost_per_query_usd": "${:.5f}",
                    "avg_latency_seconds": "{:.2f}s",
                }
            ),
            use_container_width=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                df.sort_values("avg_judge_score", ascending=False),
                x="config_name", y="avg_judge_score", color="chunking_strategy",
                title="Answer quality (LLM-judge score, 1-5) by configuration",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(
                df.sort_values("avg_recall_at_k", ascending=False),
                x="config_name", y="avg_recall_at_k", color="chunking_strategy",
                title="Retrieval recall@k by configuration",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_tradeoff:
        st.subheader("Quality vs. cost tradeoff")
        st.caption(
            "The configs worth using are the ones on the top-left frontier: "
            "high quality, low cost. Bubble size = average retrieval recall."
        )
        fig = px.scatter(
            df,
            x="avg_cost_per_query_usd",
            y="avg_judge_score",
            size="avg_recall_at_k",
            color="chunking_strategy",
            hover_name="config_name",
            text="config_name",
            title="Answer quality vs. cost per query",
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

    with tab_drilldown:
        st.subheader("Per-query results for a single configuration")
        selected_config = st.selectbox("Choose a configuration", sorted(all_results.keys()))
        queries = all_results[selected_config]["queries"]

        for qid, q in queries.items():
            score = q["judge_score"]
            score_label = f"score: {score}/5" if score is not None else "score: unparseable"
            with st.expander(f"{qid} — {score_label} — {q['question'][:80]}"):
                st.markdown(f"**Question:** {q['question']}")
                st.markdown(f"**Ground truth:** {q['ground_truth_answer']}")
                st.markdown(f"**Generated answer:** {q['generated_answer']}")
                st.markdown(f"**Judge reasoning:** {q['judge_reasoning']}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Hit@k", "Yes" if q["retrieval"]["hit_at_k"] else "No")
                c2.metric("Precision@k", f"{q['retrieval']['precision_at_k']:.1%}")
                c3.metric("Cost", f"${q['estimated_cost_usd']:.5f}")
                c4.metric("Latency", f"{q['latency_seconds']:.2f}s")


if __name__ == "__main__":
    main()
