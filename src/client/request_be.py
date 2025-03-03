import requests
from src.client.response_be import BeResponse
from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import Union, List, Optional


class BeRequest:
    def __init__(self, ip: str = "127.0.0.1", port: int = 5000, protocol: str = "http"):
        self._url = f"{protocol}://{ip}:{port}"

    def get(self, path: str) -> dict:
        response = requests.get(f"{self._url}/{path}")
        return BeResponse(response=response).json()

    def post(self, path: str, payload: Union[dict, None], pdf_file: Optional[UploadedFile] = None) -> BeResponse:
        if pdf_file is not None:
            pdf_file_parse = [("pdf_file", pdf_file)]
            response = BeResponse(response=requests.post(f"{self._url}/{path}", files=pdf_file_parse))
        elif "query" in payload.keys() :
            response = BeResponse(response=requests.post(f"{self._url}/{path}?query={payload['query']}"))
        elif "embedding_model_name" in payload.keys() and len(payload) == 1:
            response = BeResponse(response=requests.post(f"{self._url}/{path}?embedding_model_name={payload['embedding_model_name']}"))
        else:
            response = BeResponse(response=requests.post(f"{self._url}/{path}", json=payload))

        if response.is_error():
            raise RuntimeError("Something went wrong with the post request...")
        else:
            return response
