
import streamlit as st


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.caption(
            "No source metadata was returned by the API."
        )
        return

    with st.expander("View retrieved sources", expanded=False):
        for index, source in enumerate(sources, start=1):
            filename = source.get(
                "source",
                source.get("filename", "Unknown document"),
            )
            page = source.get("page")

            if page is not None:
                st.markdown(f"**{index}. {filename} — page {page}**")
            else:
                st.markdown(f"**{index}. {filename}**")

            excerpt = source.get("excerpt")
            if excerpt:
                st.caption(excerpt)
