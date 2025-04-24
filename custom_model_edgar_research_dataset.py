from io import BytesIO
from openai import AsyncOpenAI

import asyncio
from vals import Suite, Run, RunParameters

from providers_configs import provider_args
from prompts import INSTRUCTION_EDGAR_RESEARCH

def get_docs_for_prompt(files):
    files_content = [file.read().decode("utf-8") for file_name, file in files.items() if file_name.endswith(".txt")]
    doc_prompt = ""
    for i, file in enumerate(files_content):
        doc_prompt += f"Document {i+1}: {file}\n\n"

    return doc_prompt


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
        docs = get_docs_for_prompt(files)
        prompt = INSTRUCTION_EDGAR_RESEARCH.format(question=test_input, documents=docs)

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

        task = "edgar_research"
        model_under_test = "together/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
        eval_model = "openai/gpt-4o"

        parameters = RunParameters(
            eval_model=eval_model,
            temperature=0,
            max_output_tokens=1024,
            parallelism=1,
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
