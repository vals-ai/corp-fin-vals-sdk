import traceback
from collections.abc import Callable
from io import BytesIO
from math import floor
from typing import Any

from model_library.base import LLM, LLMConfig, QueryResult, TextInput, TokenRetryParams
from model_library.exceptions import MaxContextWindowExceededError
from model_library.registry_utils import get_registry_model
from model_library.utils import get_context_window_for_model

INSTRUCTION_CORP_FIN = """
I will give you a question and a document.

You need to answer the question based on the document.

--QUESTION--
{question}
--END OF QUESTION--

--DOCUMENT--
{document}
--END OF DOCUMENT--

Your answer:
"""


async def get_document_content(files: dict[str, BytesIO]) -> str:
    """
    Reads the document text
    """
    if len(files) == 1:
        document = files[list(files.keys())[0]].read().decode("utf-8")
    else:
        doc_type = "full"
        matching_key = next(
            (key for key in files if key.rsplit(".", 1)[0].endswith(doc_type)),
            None,
        )
        if matching_key is None:
            raise ValueError(f"No file found with key ending in '{doc_type}'")
        document = files[matching_key].read().decode("utf-8")
    return document


async def query_with_truncation_retry(
    llm: LLM,
    doc_text: str,
    build_prompt: Callable[[str], str],
) -> tuple[QueryResult, dict[str, int]]:
    """
    Query an LLM with automatic truncation retry on context window errors

    First truncates the document below the context window
    Then, on MaxContextWindowExceeded error, until the query suceeds:
        - Shortens the document text by 10% and retries
    """

    registry_key = llm._registry_key
    if not registry_key:
        raise ValueError("Non registry model cannot be used with truncation")
    context_window = get_context_window_for_model(model_name=registry_key)
    if not context_window:
        raise ValueError(f"Model {llm._registry_key} does not have a context window. Cannot use truncation.")

    truncation_record = {
        "initial_context_window_truncation": 0,
        "max_context_window_exceeded_error_truncation": 0,
    }

    prompt = build_prompt(doc_text)

    def shorten(shortening_ratio: float) -> str:
        new_doc_text = doc_text[: floor(len(doc_text) * shortening_ratio)]
        return build_prompt(new_doc_text)

    # first, truncate using context window
    length = await llm.count_tokens(input=[TextInput(text=prompt)])
    if length > context_window:
        truncation_record["initial_context_window_truncation"] += 1
        shorten(context_window / length)

    # shorten until query succeeds
    while True:
        try:
            return (await llm.query(prompt), truncation_record)
        except MaxContextWindowExceededError:
            # record, shorten prompt, and try again
            truncation_record["max_context_window_exceeded_error"] += 1
            prompt = shorten(0.9)


async def get_custom_model(model_name: str, parameters: dict[str, Any]):
    from vals.sdk.types import OutputObject

    if "max_output_tokens" in parameters:
        parameters["max_tokens"] = parameters.pop("max_output_tokens")
    override_config = LLMConfig.model_validate(parameters, extra="ignore")

    model = get_registry_model(
        model_name,
        override_config=override_config,
    )

    token_retry_params = parameters.get("token_retry_params", None)
    if token_retry_params:
        await model.init_token_retry(
            token_retry_params=TokenRetryParams.model_validate(token_retry_params),
        )

    async def custom_call(test_input: str, files: dict[str, BytesIO], context: dict[str, Any]):
        try:
            # build prompt
            doc_content = await get_document_content(files)

            def build_prompt(document_text: str):
                return INSTRUCTION_CORP_FIN.format(question=test_input, document=document_text)

            # query
            query_result, truncation_record = await query_with_truncation_retry(
                llm=model, doc_text=doc_content, build_prompt=build_prompt
            )

            # build output object
            context = {}
            context["truncation_record"] = truncation_record
            query_result.metadata.extra
            if query_result.reasoning:
                context["reasoning"] = query_result.reasoning

            return OutputObject(
                llm_output=query_result.output_text_str,
                in_tokens=query_result.metadata.total_input_tokens,
                out_tokens=query_result.metadata.total_output_tokens,
                duration=query_result.metadata.duration_seconds,
                output_context=context,
            )
        except Exception as e:
            print(f"Error querying custom model: {e}")
            traceback.print_exc()
            raise e

    return custom_call
