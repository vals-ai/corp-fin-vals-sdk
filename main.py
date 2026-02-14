import asyncio
import json
import logging

import click
from custom_model import get_custom_model
from dotenv import load_dotenv
from model_library import set_logging
from vals import Run, RunParameters, Suite


async def main(
    model_under_test: str,
    eval_model: str,
    resume_run_id: str | None,
    task: str,
    parallelism: int,
    max_tokens: int,
    temperature: float | None,
):
    parameters = RunParameters(
        model_under_test="corpfin",
        eval_model=eval_model,
        parallelism=parallelism,
        max_output_tokens=max_tokens,
        temperature=temperature,
    )

    with open("suites.json") as f:
        mapping_suite_ids: dict[str, str] = json.load(f)

    if "REPLACE_ME" in mapping_suite_ids.values():
        raise ValueError("Please replace the suite ids in suites.json with the suite IDs provided in the email.")

    custom_model = await get_custom_model(
        model_name=model_under_test,
        parameters=parameters.model_dump(),
    )

    if resume_run_id:
        # resume a run
        run = await Run.from_id(resume_run_id)
        await run.resume_run(model=custom_model)
        await run.refresh()
    else:
        # start a new run
        task_suite_id = mapping_suite_ids[task]
        suite = await Suite.from_id(task_suite_id)
        run = await suite.run(
            model=custom_model,
            model_name=model_under_test,
            parameters=parameters,
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
    help="The model under test",
)
@click.option(
    "--eval_model",
    default="anthropic/claude-sonnet-4-5-20250929",
    help="The model to use as the evaluator (grader)",
)
@click.option(
    "--resume_run_id",
    default=None,
    required=False,
    help="Resume a previous run from its ID",
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
    help="The number of concurrent questions to run simultaneously",
)
@click.option(
    "--max_tokens",
    default=32000,
    required=False,
    help="The max tokens to use for the model",
)
@click.option(
    "--temperature",
    default=None,
    required=False,
    help="The temperature to use for the model",
)
def cli(*args, **kwargs):
    load_dotenv(override=True)
    set_logging(True, level=logging.WARNING)
    asyncio.run(main(*args, **kwargs))


if __name__ == "__main__":
    cli()
