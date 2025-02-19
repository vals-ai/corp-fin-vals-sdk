import asyncio
import time
from io import BytesIO
from typing import Any

from vals import Check, QuestionAnswerPair, Run, RunParameters, Suite, Test
from vals.sdk.types import OperatorInput, OperatorOutput


async def run_with_function():
    """Run the suite on a custom model function."""
    suite = await Suite.from_id("") # put your suite id here

    def function(input_under_test: str) -> str:
        # This would be replaced with your custom model.
        return input_under_test + "!!!"

    def function_with_context_and_files(
        input_under_test: str, files: dict[str, BytesIO], context: dict[str, Any]
    ) -> str:
        # Your LLM would leverage the context, the files, and the input_under_test
        # to return a response.
        return input_under_test + " with context!!!"

    run = await suite.run(
        model=function, wait_for_completion=True, model_name="my_custom_model"
    )

    print(f"Run URL: {run.url}")
    print(f"Pass percentage: {run.pass_percentage}")


if __name__ == "__main__":
    asyncio.run(run_with_function())
