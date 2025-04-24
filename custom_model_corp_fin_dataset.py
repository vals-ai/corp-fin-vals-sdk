from io import BytesIO
import tiktoken
from openai import AsyncOpenAI
import google
from google.genai import types
import google.generativeai as genai

# import google.generativeai as genai
import os
import asyncio
from vals import Suite, Run, RunParameters

provider_args = {
    "openai": {"api_key": os.getenv("OPENAI_API_KEY")},
    "google": {
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "anthropic": {
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "base_url": "https://api.anthropic.com/v1/",
    },
    "together": {
        "api_key": os.getenv("TOGETHER_API_KEY"),
        "base_url": "https://api.together.xyz/v1/",
    },
    "fireworks": {
        "api_key": os.getenv("FIREWORKS_API_KEY"),
        "base_url": "https://api.fireworks.ai/v1/",
    },
    "mistralai": {
        "api_key": os.getenv("MISTRAL_API_KEY"),
        "base_url": "https://api.mistral.ai/v1/",
    },
    "grok": {
        "api_key": os.getenv("GROK_API_KEY"),
        "base_url": "https://api.x.ai/v1",
    },
    "cohere": {
        "api_key": os.getenv("COHERE_API_KEY"),
        "base_url": "https://api.cohere.ai/compatibility/v1/",
    },
}

INSTRUCTION = """
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


def get_doc_type(model_name):
    if "gpt-4o" or "Llama-3.3-70B-Instruct-Turbo" in model_name:
        return "trimmed_openai"
    elif "claude" in model_name:
        return "trimmed_anthropic"
    elif "gemini" in model_name:
        return "full"
    else:
        raise ValueError(f"Model {model_name} not supported")


def get_doc(model_name, files):
    if len(files) == 1:
        return files[list(files.keys())[0]].read().decode("utf-8")
    doc_type = get_doc_type(model_name)

    matching_key = next(
        (key for key in files.keys() if key.rsplit(".", 1)[0].endswith(doc_type)), None
    )

    if matching_key is None:
        raise ValueError(f"No file found with key ending in '{doc_type}'")

    document = files[matching_key].read().decode("utf-8")

    if "gpt-4o" in model_name:
        encoding = tiktoken.encoding_for_model("gpt-4o")
        tokens = encoding.encode(document)[:125000]
        return encoding.decode(tokens)

    return document


async def call_model(
    client, model_key, prompt, temperature, max_tokens, system_prompt=None
):
    messages = []
    if system_prompt is not None and len(system_prompt) > 0:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    return await client.chat.completions.create(
        model=model_key,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def read_output(output):
    return {
        "llm_output": output.choices[0].message.content,
        "metadata": {
            "in_tokens": output.usage.prompt_tokens,
            "out_tokens": output.usage.completion_tokens,
        },
    }


def get_custom_model(model_name, parameters, *args, **kwargs):
    system_prompt = parameters.get("system_prompt", "")
    temperature = parameters.get("temperature", 0)
    max_tokens = parameters.get("max_output_tokens", 512)
    provider, model_key = model_name.split("/", 1)
    client = AsyncOpenAI(**provider_args[provider])

    async def custom_call(
        test_input: str, files: dict[str, BytesIO], context: dict[str, any]
    ):
        doc = get_doc(model_name, files)
        prompt = INSTRUCTION.format(question=test_input, document=doc)

        try:
            output = await call_model(
                client, model_key, prompt, temperature, max_tokens, system_prompt
            )
            out = read_output(output)
            return out
        except Exception as e:
            print(e)
            return "error when calling model"

    return custom_call


if __name__ == "__main__":
    async def main():
        # Replace the mapping_suite_ids with the mapping we provided you by email.
        mapping_suite_ids = {}

        task = "shared_max_context"
        model_under_test = "together/meta-llama/Llama-3.3-70B-Instruct-Turbo"
        eval_model = "anthropic/claude-3-5-sonnet-20241022"

        parameters = RunParameters(
            eval_model=eval_model,
            temperature=0,
            max_output_tokens=1024,
            parallelism=2,
        )

        custom_model = get_custom_model(model_under_test, parameters.model_dump())

        suite = await Suite.from_id(mapping_suite_ids[task])
        await suite.run(
            model=custom_model,
            model_name=model_under_test,
            parameters=parameters,
            upload_concurrency=3,
        )

        # To resume a run that crashed
        # run = await Run.from_id("run_id")
        # await run.resume_run(model=custom_model)

    asyncio.run(main())
