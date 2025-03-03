import streamlit as st

from loguru import logger
from src.client.request_be import BeRequest
from time import time

class HaystackFrontend:
    request_be: BeRequest
    prompt_template: str = """You are a successful scientist and have read the book "Columnar Structures of Spheres: Fundamentals and Applications" by Jens Winkelmann and Ho-Kei Chan. You are now asked to answer questions based on the given context. The context are paragraphs from the book that fit best to the question."""
    prompt_addition: str = " Write the answer as markdown text."

    def __init__(self, ip: str = "127.0.0.1", port: int = 8000, protocol: str = "http") -> None:
        self._request_be = BeRequest(ip, port, protocol)

        st.session_state["index_pipeline"] = st.session_state.get("index_pipeline", None)
        st.session_state["query_pipeline"] = st.session_state.get("query_pipeline", None)

        st.session_state["uploaded_images"] = st.session_state.get("uploaded_images", [])
        st.session_state["response_text"] = st.session_state.get("response_text", None)

    def build_pipeline_widget(self) -> None:
        with st.expander("Build pipeline", expanded=False):
            embedding_model_name = st.radio("Select the embedding model", self._request_be.get("get_embedding_models"))
            llm_model_name = st.radio("Select the LLM model", self._request_be.get("get_llm_models"))
            prompt_template = st.text_area("Enter the prompt template", value=self.prompt_template, height=150)

            pipeline_button = st.button("Build pipeline")

            if pipeline_button:
                with st.spinner("Building index pipeline..."):
                    self._request_be.post("build_index_pipeline", payload={"embedding_model_name": embedding_model_name})
                    st.session_state["index_pipeline"] = True
                    st.success("Index pipeline built successfully.")

                    self._request_be.post("build_query_pipeline", payload={
                        "llm_model_name": llm_model_name,
                        "embedding_model_name": embedding_model_name,
                        "prompt_template": prompt_template + self.prompt_addition
                    })
                    st.session_state["query_pipeline"] = True
                    st.success("Query pipeline built successfully.")

    def build_upload_widget(self) -> None:
        with st.form("Upload_widget", clear_on_submit=True):
            uploaded_images = st.file_uploader("Upload your pdf file here...")
            upload_button = st.form_submit_button("Upload pdf")

            if uploaded_images is not None and \
                    (st.session_state["index_pipeline"] is None or st.session_state["query_pipeline"] is None):
                st.error("Please build the pipeline first.")
                return

            if uploaded_images is not None and upload_button:
                with st.spinner("Uploading pdf..."):
                    success = self._request_be.post("run_index_pipeline", None, pdf_file=uploaded_images)
                    logger.info(f"Upload successful: {success}.")

                    st.session_state["uploaded_images"] = [uploaded_images.name]

            if len(st.session_state["uploaded_images"]) > 0:
                with st.container(border=True):
                    uploaded_images_str = "\n".join([f"- {image}" for image in st.session_state["uploaded_images"]])
                st.success(f"Successfully uploaded the following files:\n{uploaded_images_str}.")


    def build_page(self) -> None:
        self.build_pipeline_widget()
        self.build_upload_widget()
        self.build_run_haystack()

    def build_run_haystack(self) -> None:
        with st.container(border=True):
            if len(st.session_state["uploaded_images"]) > 0:
                query = st.text_area("Ask a question based on the context of the uploaded pdf file.")
                query_button = st.button("Ask question")

                if query_button:
                    with st.spinner("Querying Haystack..."):
                        t0 = time()
                        response = self._request_be.post("query_pipeline", payload={"query": query})
                        st.session_state["response_text"] = response.text().replace(r"\n", "  \n")
                        logger.info("Querying Haystack took {:.2f}s.".format(time() - t0))

        with st.container(border=True):
            if st.session_state["response_text"] is not None:
                st.markdown(st.session_state["response_text"])


ai_ocr_frontend = HaystackFrontend()
ai_ocr_frontend.build_page()
