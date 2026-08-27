import streamlit as st
from pipeline import run_search_pipeline

st.set_page_config(page_title="Multi-Agent Research System", page_icon="🔎", layout="wide")

st.title("🔎 Multi-Agent Research System")
st.caption("Search Agent → Reader Agent → Writer → Critic")

# ---- Input ----
with st.form("topic_form"):
    topic = st.text_input("Enter a research topic", placeholder="e.g. Impact of AI on climate modeling")
    submitted = st.form_submit_button("Run Research Pipeline")

if "state" not in st.session_state:
    st.session_state.state = None

# ---- Run pipeline ----
if submitted:
    if not topic.strip():
        st.warning("Please enter a topic before running the pipeline.")
    else:
        state = {}
        status = st.status("Starting pipeline...", expanded=True)

        try:
            # Step 1: Search Agent
            status.update(label="Step 1/4 — Search agent gathering information...")
            from agent import build_search_agent, build_reader_agent, writer_chain, critic_chain

            search_agent = build_search_agent()
            search_result = search_agent.invoke({
                "messages": [("user", f"Find recent reliable and detailed information about: {topic}")]
            })
            state["search_result"] = search_result["messages"][-1].content
            status.write("✅ Search complete")
            with st.expander("Search Result", expanded=False):
                st.write(state["search_result"])

            # Step 2: Reader Agent
            status.update(label="Step 2/4 — Reader agent scraping top resource...")
            reader_agent = build_reader_agent()
            reader_result = reader_agent.invoke({
                "messages": [("user",
                    f"Based on the following search result about '{topic}' "
                    f"pick the most relevant URL and scrape it for deeper content.\n\n"
                    f"Search Result:\n{state['search_result'][:800]}"
                )]
            })
            state["scraped_content"] = reader_result["messages"][-1].content
            status.write("✅ Scraping complete")
            with st.expander("Scraped Content", expanded=False):
                st.write(state["scraped_content"])

            # Step 3: Writer
            status.update(label="Step 3/4 — Writer drafting the report...")
            research_combined = (
                f"SEARCH RESULT:\n{state['search_result']}\n\n"
                f"DETAILED SCRAPED RESULT:\n{state['scraped_content']}"
            )
            state["report"] = writer_chain.invoke({
                "topic": topic,
                "research": research_combined
            })
            status.write("✅ Draft report ready")

            # Step 4: Critic
            status.update(label="Step 4/4 — Critic reviewing the report...")
            state["feedback"] = critic_chain.invoke({
                "report": state["report"]
            })
            status.update(label="Pipeline complete ✅", state="complete")

            st.session_state.state = state

        except Exception as e:
            status.update(label="Pipeline failed ❌", state="error")
            st.exception(e)

# ---- Display results ----
state = st.session_state.state
if state:
    st.divider()
    st.subheader(f"📄 Final Report")
    st.markdown(state["report"] if isinstance(state["report"], str) else str(state["report"]))

    st.subheader("🧐 Critic Feedback")
    st.markdown(state["feedback"] if isinstance(state["feedback"], str) else str(state["feedback"]))

    with st.expander("Raw research data (search + scraped content)"):
        st.markdown("**Search Result**")
        st.write(state["search_result"])
        st.markdown("**Scraped Content**")
        st.write(state["scraped_content"])

    st.download_button(
        "Download Report as .txt",
        data=str(state["report"]),
        file_name="research_report.txt",
        mime="text/plain",
    )
