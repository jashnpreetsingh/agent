"""Streamlit interface.

    streamlit run src/ui/app.py

Shows the answer alongside the reasoning trace, because the interesting part of
an agent is *how* it reached an answer - and because a citation the user cannot
click is a citation they cannot check.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st

# Allow `streamlit run src/ui/app.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent import PubMedAgent  # noqa: E402
from src.config import TransportMode, get_settings  # noqa: E402
from src.llm.errors import LLMError  # noqa: E402
from src.memory.store import ConversationStore  # noqa: E402
from src.tools.pubmed import PubMedError  # noqa: E402

st.set_page_config(page_title="PubMed Evidence Agent", page_icon="🔬", layout="wide")

GRADE_COLOURS = {
    "strong": "#1a7f37",
    "moderate": "#9a6700",
    "limited": "#bc4c00",
    "insufficient": "#82071e",
}


@st.cache_resource
def get_agent() -> PubMedAgent:
    settings = get_settings()
    return PubMedAgent(settings, store=ConversationStore(settings.memory_db))


def init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"ui-{uuid.uuid4().hex[:8]}"
    if "history" not in st.session_state:
        st.session_state.history = []


def sidebar() -> dict:
    settings = get_settings()
    with st.sidebar:
        st.header("Configuration")
        st.caption(f"**Model:** `{settings.model_name}`")
        st.caption(f"**Embeddings:** `{settings.embedding_model}`")
        st.caption(f"**LLM mode:** `{settings.llm_mode.value}`")
        st.caption(f"**PubMed mode:** `{settings.pubmed_mode.value}`")

        if settings.llm_mode is not TransportMode.REPLAY and not settings.gemini_api_key:
            st.error("No GEMINI_API_KEY set. Add one to .env, or use LLM_MODE=replay.")

        st.divider()
        st.header("Session")
        st.caption(f"`{st.session_state.session_id}`")
        use_memory = st.toggle(
            "Remember context", value=True, help="Carry earlier turns into follow-up questions."
        )
        show_trace = st.toggle("Show reasoning trace", value=True)

        if st.button("New session", use_container_width=True):
            st.session_state.session_id = f"ui-{uuid.uuid4().hex[:8]}"
            st.session_state.history = []
            st.rerun()

        st.divider()
        st.caption(
            "Research summaries of published literature. Not medical advice — "
            "consult a qualified clinician for individual care."
        )
    return {"use_memory": use_memory, "show_trace": show_trace}


def render_answer(answer, tracer, show_trace: bool) -> None:
    if not answer.answered:
        st.warning(answer.summary)
        st.caption(f"Category: `{answer.refusal_category.value if answer.refusal_category else 'blocked'}`")
        return

    grade = answer.overall_grade.value
    st.markdown(
        f"<span style='background:{GRADE_COLOURS.get(grade, '#555')};color:white;"
        f"padding:2px 10px;border-radius:10px;font-size:0.8em'>evidence: {grade}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"### {answer.summary}")

    if answer.claims:
        st.markdown("#### Findings")
        for claim in answer.claims:
            links = " ".join(
                f"[{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)" for pmid in claim.pmids
            )
            st.markdown(f"- {claim.statement}  \n  <sub>{links} · {claim.confidence.value}</sub>",
                        unsafe_allow_html=True)

    if answer.limitations:
        with st.expander("Limitations", expanded=False):
            for item in answer.limitations:
                st.markdown(f"- {item}")

    if answer.citations:
        with st.expander(f"Sources ({len(answer.citations)})", expanded=False):
            for article in answer.citations:
                st.markdown(
                    f"**[{article.title}]({article.url})**  \n"
                    f"<sub>{article.citation()} · "
                    f"{', '.join(article.publication_types[:3]) or 'unclassified'}</sub>",
                    unsafe_allow_html=True,
                )
                if article.abstract:
                    st.caption(article.abstract[:320] + ("…" if len(article.abstract) > 320 else ""))

    audit = answer.citation_audit
    summary = tracer.summary()
    cols = st.columns(5)
    cols[0].metric("Claims cited", f"{audit.cited_claims}/{audit.total_claims}")
    cols[1].metric("Fabricated PMIDs", len(audit.hallucinated_pmids))
    cols[2].metric("LLM calls", summary["llm_requests"])
    cols[3].metric("Tool calls", summary["tool_calls"])
    cols[4].metric("Time", f"{answer.elapsed_seconds}s")

    if answer.degraded:
        st.info("This answer used a fallback path — see the trace for details.")
    for note in answer.notes:
        st.caption(f"note: {note}")

    if show_trace:
        with st.expander("Reasoning trace", expanded=False):
            for event in tracer.events:
                if event.name.endswith(".start"):
                    continue
                timing = f" · {event.duration_ms:.0f} ms" if event.duration_ms else ""
                st.markdown(f"**`{event.name}`**{timing}")
                payload = {k: v for k, v in event.data.items() if k != "ok" and v not in (None, "", [], {})}
                if payload:
                    st.json(payload, expanded=False)


def main() -> None:
    init_state()
    options = sidebar()

    st.title("🔬 PubMed Evidence Agent")
    st.caption(
        "Ask a biomedical question. The agent plans searches, queries PubMed, ranks the "
        "evidence, and answers with a citation on every claim."
    )

    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            render_answer(entry["answer"], entry["tracer"], options["show_trace"])

    if question := st.chat_input("e.g. Do SGLT2 inhibitors reduce heart failure hospitalisation?"):
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.status("Researching…", expanded=True) as status:
                try:
                    status.write("Planning searches, querying PubMed, ranking evidence…")
                    result = get_agent().run(
                        question,
                        session_id=st.session_state.session_id if options["use_memory"] else None,
                    )
                    status.update(label="Done", state="complete", expanded=False)
                except (LLMError, PubMedError, ValueError) as exc:
                    status.update(label="Failed", state="error")
                    st.error(f"The run could not complete: {exc}")
                    return

            render_answer(result.answer, result.tracer, options["show_trace"])
            st.session_state.history.append(
                {"question": question, "answer": result.answer, "tracer": result.tracer}
            )


if __name__ == "__main__":
    main()
