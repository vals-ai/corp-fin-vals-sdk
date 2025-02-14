from io import BytesIO
import tiktoken
from openai import AsyncOpenAI
import google
import os
import asyncio
from vals import Suite, Run, RunParameters

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
google_client = google.genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

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
    if "gpt-4o" in model_name:
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
        tokens = encoding.encode(document)[:120000]
        return encoding.decode(tokens)

    return document


async def call_model(
    model_key, provider, prompt, temperature, max_tokens, system_prompt=None
):
    if provider == "google":
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_prompt is not None and len(system_prompt) > 0:
            generation_config["system_instruction"] = system_prompt

        return await google_client.aio.models.generate_content(
            model=model_key,
            contents=[prompt],
            generation_config=generation_config,
        )

    elif provider == "openai":
        messages = []
        if system_prompt is not None and len(system_prompt) > 0:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return await openai_client.chat.completions.create(
            model=model_key,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    else:
        raise ValueError(f"Provider {provider} not supported")


def read_output(output):
    return {
        "llm_output": output.choices[0].message.content,
        "metadata": {
            "in_tokens": output.usage.prompt_tokens,
            "out_tokens": output.usage.completion_tokens,
        },
    }


async def custom_model(model_name, parameters, *args, **kwargs):
    system_prompt = parameters.get("system_prompt", "")
    temperature = parameters.get("temperature", 0)
    max_tokens = parameters.get("max_output_tokens", 512)

    async def custom_call(
        test_input: str, files: dict[str, BytesIO], context: dict[str, any]
    ):
        model_key, provider = model_name.split("/")

        doc = get_doc(model_name, files)
        prompt = INSTRUCTION.format(question=test_input, document=doc)

        try:
            output = await call_model(
                model_key, provider, prompt, temperature, max_tokens, system_prompt
            )
            out = read_output(output)
            return out
        except Exception as e:
            print(e)
            return "error when calling model"

    return custom_call


if __name__ == "__main__":

    async def main():
        mapping_suite_ids = {
            "exact_pages_task": "corp_fin_v2",
            "shared_max_context_task": "corp_fin_v2_diverse",
            "max_fitting_context_task": "corp_fin_v2_max_fitting_context",
        }

        model_under_test = "google/gemini-2.0-flash-001"
        eval_model = "openai/gpt-4o"

        parameters = RunParameters(
            eval_model=eval_model,
            system_prompt="You are a helpful assistant.",
            temperature=0,
            max_output_tokens=512,
            parallelism=5,
        )

        custom_model = custom_model(model_under_test, parameters.model_dump())

        suite = await Suite.from_id("corp_fin_v2")
        await suite.run(model=custom_model, parameters=parameters, upload_concurrency=5)

        # To resume a run that crashed
        # run = await Run.from_id("run_id")
        # await run.resume_run(model=custom_model, parameters=parameters, upload_concurrency=5)

    asyncio.run(main())
