"""Gradio frontend for GoFetch RAG system."""

import json
import os

import gradio as gr
import httpx

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def _format_metadata(meta: dict) -> tuple[str, str]:  # noqa: C901
    """Format citation and latency metadata from SSE response.

    Args:
        meta: Parsed metadata dictionary from the SSE stream.

    Returns:
        Tuple of (citations_text, latency_text).
    """
    citations = meta.get("citations", [])
    latency = meta.get("latency_ms", {})
    confidence = meta.get("confidence", 0)
    low_conf = meta.get("low_confidence", False)

    cite_lines = []
    for c in citations:
        cite_lines.append(
            f"[{c['index']}] {c['source']} (score: {c['score']:.4f})\n   {c['text'][:150]}..."
        )
    citations_text = "\n\n".join(cite_lines)

    lat_lines = [f"  {k}: {v:.1f}ms" for k, v in latency.items()]
    latency_text = "\n".join(lat_lines)
    conf_label = "(LOW)" if low_conf else ""
    latency_text += f"\n  confidence: {confidence:.4f} {conf_label}".rstrip()

    return citations_text, latency_text


def _parse_sse_stream(response: httpx.Response) -> tuple[str, str, str]:
    """Parse an SSE stream into answer, citations, and latency text.

    Args:
        response: Streaming httpx response.

    Returns:
        Tuple of (answer, citations_text, latency_text).
    """
    answer = ""
    citations_text = ""
    latency_text = ""
    event_type = ""
    buffer = ""

    for chunk in response.iter_text():
        buffer += chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()

            if line.startswith("event:"):
                event_type = line[len("event:") :].strip()
                continue

            if not line.startswith("data:"):
                continue

            data = line[len("data:") :].strip()
            if event_type == "token":
                answer += data
            elif event_type == "metadata":
                try:
                    meta = json.loads(data)
                    citations_text, latency_text = _format_metadata(meta)
                except json.JSONDecodeError:
                    pass

    return answer, citations_text, latency_text


def stream_query(query: str) -> str:
    """Stream a query to the backend and return formatted answer.

    Args:
        query: The user's question.

    Returns:
        The complete answer text with citations and latency.
    """
    if not query.strip():
        return "Please enter a question."

    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream("GET", f"{BACKEND_URL}/query", params={"q": query}) as response:
                answer, citations_text, latency_text = _parse_sse_stream(response)
    except httpx.ConnectError:
        return "Error: Cannot connect to backend. Is the server running?"
    except Exception as exc:
        return f"Error: {exc}"

    result = answer
    if citations_text:
        result += f"\n\n---\n**Sources:**\n\n{citations_text}"
    if latency_text:
        result += f"\n\n---\n**Latency:**\n{latency_text}"

    return result


def upload_files(files: list[str]) -> str:
    """Upload files to the backend for ingestion.

    Args:
        files: List of file paths from Gradio upload.

    Returns:
        Status message.
    """
    if not files:
        return "No files selected."

    upload_files_list: list[tuple[str, object]] = []
    try:
        for file_path in files:
            upload_files_list.append(("files", open(file_path, "rb")))

        with httpx.Client(timeout=300.0) as client:
            response = client.post(f"{BACKEND_URL}/ingest", files=upload_files_list)

        if response.status_code == 200:
            data = response.json()
            return f"Ingested {data['documents']} documents into {data['chunks']} chunks."
        return f"Ingestion failed: {response.text}"

    except httpx.ConnectError:
        return "Error: Cannot connect to backend. Is the server running?"
    except Exception as exc:
        return f"Error: {exc}"
    finally:
        for _, fh in upload_files_list:
            fh.close()


def ingest_data_dir() -> str:
    """Trigger ingestion of all files in the data/ directory.

    Returns:
        Status message.
    """
    try:
        with httpx.Client(timeout=300.0) as client:
            response = client.post(f"{BACKEND_URL}/ingest")

        if response.status_code == 200:
            data = response.json()
            return f"Ingested {data['documents']} documents into {data['chunks']} chunks."
        return f"Ingestion failed: {response.text}"

    except httpx.ConnectError:
        return "Error: Cannot connect to backend. Is the server running?"
    except Exception as exc:
        return f"Error: {exc}"


# --- Gradio UI ---

with gr.Blocks(title="GoFetch RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# GoFetch RAG System")
    gr.Markdown(
        "Ask questions about your documents. "
        "Answers include inline citations [1][2] linked to source chunks."
    )

    with gr.Tab("Query"):
        query_input = gr.Textbox(
            label="Question",
            placeholder="What is the attention mechanism?",
            lines=2,
        )
        query_btn = gr.Button("Ask", variant="primary")
        answer_output = gr.Markdown(label="Answer")

        query_btn.click(fn=stream_query, inputs=query_input, outputs=answer_output)
        query_input.submit(fn=stream_query, inputs=query_input, outputs=answer_output)

    with gr.Tab("Ingest"):
        gr.Markdown("Upload documents or ingest from the `data/` directory.")

        file_upload = gr.File(
            label="Upload Documents",
            file_count="multiple",
            file_types=[".pdf", ".txt"],
        )
        upload_btn = gr.Button("Upload & Ingest")
        upload_status = gr.Textbox(label="Status", interactive=False)

        upload_btn.click(fn=upload_files, inputs=file_upload, outputs=upload_status)

        gr.Markdown("---")
        ingest_btn = gr.Button("Ingest from data/ directory")
        ingest_status = gr.Textbox(label="Status", interactive=False)
        ingest_btn.click(fn=ingest_data_dir, outputs=ingest_status)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
