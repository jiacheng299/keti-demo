import streamlit as st


def main() -> None:
    """Render the current application shell."""
    st.set_page_config(page_title="开放场景视觉语义认知 Demo", layout="wide")
    st.title("开放场景视觉语义认知 Demo")
    st.markdown("当前阶段：项目骨架。")


if __name__ == "__main__":
    main()
