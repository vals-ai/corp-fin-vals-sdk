import traceback
from io import BytesIO
from typing import Any

from model_library.base import LLMConfig, TokenRetryParams
from model_library.query_utils import query_with_truncation_retry
from model_library.registry_utils import get_registry_model

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

    async def custom_call(test_input: str, files: dict[str, BytesIO], context: dict[str, Any], question_id: str, run_id: str):
        try:
            # build prompt
            doc_content = await get_document_content(files)

            def build_prompt(document_text: str):
                return INSTRUCTION_CORP_FIN.format(question=test_input, document=document_text)

            # query
            query_result, truncation_record = await query_with_truncation_retry(
                llm=model, doc_text=doc_content, build_prompt=build_prompt,
                question_id=question_id, run_id=run_id,
            )

            # build output object
            output_context = {**context, **query_result.metadata.extra, "truncation_record": truncation_record}
            if query_result.reasoning:
                output_context["reasoning"] = query_result.reasoning

            return OutputObject(
                llm_output=query_result.output_text_str,
                in_tokens=query_result.metadata.in_tokens,
                out_tokens=query_result.metadata.out_tokens,
                reasoning_tokens=query_result.metadata.reasoning_tokens,
                cache_read_tokens=query_result.metadata.cache_read_tokens,
                cache_write_tokens=query_result.metadata.cache_write_tokens,
                duration=query_result.metadata.duration_seconds,
                cost=query_result.metadata.cost.total if query_result.metadata.cost else None,
                output_context=output_context,
            )
        except Exception as e:
            print(f"Error querying custom model: {e}")
            traceback.print_exc()
            raise e

    return custom_call
