import traceback
import logging
import tiktoken
import click
import asyncio
import json

from prompts import INSTRUCTION_CORP_FIN
from io import BytesIO
from typing import Any
from model_library.base import LLMConfig, QueryResult
from model_library.registry_utils import get_max_document_tokens, get_registry_model
from model_library.logging import set_logging
from vals.sdk.types import OutputObject
from vals import Suite, RunParameters


set_logging(False)


def create_override_config(**kwargs) -> LLMConfig:
    # Filter kwargs to only include valid LLMConfig fields
    valid_kwargs = {k: v for k, v in kwargs.items() if k in LLMConfig.model_fields}

    # hardcode fix for max output tokens
    if "max_output_tokens" in kwargs and "max_tokens" not in kwargs:
        valid_kwargs["max_tokens"] = kwargs["max_output_tokens"]

    return LLMConfig(**valid_kwargs)


def query_result_to_output_object(query_result: QueryResult) -> OutputObject:
    output_context = {}

    if query_result.reasoning:
        output_context["reasoning"] = query_result.reasoning

    in_tokens = (
        query_result.metadata.in_tokens + query_result.metadata.cache_read_tokens
        or 0 + query_result.metadata.cache_write_tokens
        or 0
    )
    out_tokens = (
        query_result.metadata.out_tokens + query_result.metadata.cache_read_tokens or 0
    )

    return OutputObject(
        llm_output=query_result.output_text or "",
        in_tokens=in_tokens,
        out_tokens=out_tokens,
        duration=query_result.metadata.duration_seconds,
        output_context=output_context,
    )


def _contains_any(text: str, *needles: str) -> bool:
    lower_text = text.lower()
    return any(needle.lower() in lower_text for needle in needles)


def _trim_with_tiktoken(
    document: str, max_tokens: int, model_for_encoding: str = "gpt-4o"
) -> str:
    encoding = tiktoken.encoding_for_model(model_for_encoding)
    original_tokens = encoding.encode(document)
    tokens = original_tokens[:max_tokens]
    return encoding.decode(tokens)


def get_doc(model_name: str, files: dict[str, BytesIO], max_tokens: int) -> str:
    # Read the document based on available files and desired doc type
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

    return _trim_with_tiktoken(
        document=document,
        max_tokens=max_tokens,
        model_for_encoding="gpt-4o",
    )


def get_custom_model(model_name: str, parameters: dict[str, Any], output_buffer: int):
    max_tokens = get_max_document_tokens(model_name, output_buffer=output_buffer)

    # Keep the existing call path as-is to preserve behavior
    model = get_registry_model(model_name, create_override_config(**parameters))

    async def custom_call(
        test_input: str, files: dict[str, BytesIO], context: dict[str, Any]
    ):
        try:
            doc = get_doc(model_name, files, max_tokens)
            prompt = INSTRUCTION_CORP_FIN.format(question=test_input, document=doc)
            query_result = await model.query(prompt)
            return query_result_to_output_object(query_result)
        except Exception as e:
            print(f"Error querying custom model: {e}")
            traceback.print_exc()
            raise e

    return custom_call


async def run_async(
    model_under_test: str,
    eval_model: str,
    task: str,
    output_buffer: int,
    parallelism: int,
    temperature: float,
):
    parameters = RunParameters(
        eval_model=eval_model,
        temperature=temperature,
        parallelism=parallelism,
        max_output_tokens=32000,
    )

    with open("suites.json", "r") as f:
        mapping_suite_ids = json.load(f)

    if "REPLACE_ME" in mapping_suite_ids.values():
        raise ValueError(
            "Please replace the suite ids in suites.json with the suite IDs provided in the email."
        )

    custom_model = get_custom_model(
        model_under_test, parameters.model_dump(), output_buffer
    )

    suite = await Suite.from_id(mapping_suite_ids[task])

    # To resume a run that crashed
    # run = await Run.from_id("run_id")
    # await run.resume_run(model=custom_model)

    run = await suite.run(
        model=custom_model,
        model_name=model_under_test,
        parameters=parameters,
        upload_concurrency=10,
        wait_for_completion=True,
    )

    print("Finished running benchmark")
    print("Run ID: ", run.id)
    print("Pass Rate: ", run.pass_rate)
    print("URL: ", run.url)


@click.command()
@click.option(
    "--model_under_test",
    default="anthropic/claude-sonnet-4-5-20250929-thinking",
    required=True,
    help="Name of the model under test",
)
@click.option(
    "--eval_model",
    default="anthropic/claude-sonnet-4-5-20250929",
    help="The model to use as the evaluator (grader) for the benchmark.",
)
@click.option(
    "--output-buffer",
    default=40000,
    required=False,
    help="For max context, this is the amount of tokens to subtract from the context window to avoid context window errors.",
)
@click.option(
    "--task",
    default="shared_max_context",
    required=True,
    help="Task to run",
    type=click.Choice(
        ["shared_max_context", "max_fitting_context", "exact_pages"],
        case_sensitive=True,
    ),
)
@click.option(
    "--parallelism",
    default=10,
    required=False,
    help="The number of concurrent questions to run simultaneously.",
)
@click.option(
    "--temperature",
    default=1,
    required=False,
    help="The temperature to use for the model.",
)
def run(*args, **kwargs):
    asyncio.run(run_async(*args, **kwargs))


if __name__ == "__main__":
    run()
