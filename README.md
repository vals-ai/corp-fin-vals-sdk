# Running Corp Fin Benchmark

Our CorpFin benchmark is run via SDK because it requires different parsing of the files depending on the model's context window for one task, which makes it easier to handle everything from code.

There are three main tasks in the benchmark. All have the same data, but they pass the document context to the model in different ways:

- _Exact Pages_: Passes only the exact pages in the document needed to answer the question (generally one to two pages)
- _Shared Max Context_: Passes ~128,000 tokens worth of the document. The selection is guaranteed to have the information needed, and is independent of the model being evaluated.
- _Max Fitting Context_: Passes as much of the document as will fit in the context window of the model, starting from the beginning of the document.

For more details on the benchmark, please refer to the our [public website](https://www.vals.ai/benchmarks/corp_fin_v2).

## Set up

### Dependencies

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management. Then run:

```
make install
source .venv/bin/activate
```

### Platform

Make an account on [platform.vals.ai](https://www.platform.vals.ai/auth) with your company email address. Go to the admin page and create a new API key for yourself.

### Environment Variables

Create a `.env` file in the root of the project and add the following:

```
VALS_API_KEY=<api_key>
ANTHROPIC_API_KEY=<anthropic_api_key>
OPENAI_API_KEY=<openai_api_key>
ETC_API_KEY=<etc_api_key>
```

The `.env` takes precedence over set environment variables.

Finally, you should add the "Test Suite IDs" to suites.json. These should have generally been provided to you via email, but you can also find them in the platform, by navigating to the "Test Suites" page, clicking the relevant test suite, and looking on the right sidebar under "Test Suite ID".

## Running the benchmark

For a list of command line options, run `python main.py --help`

To run, for example, the shared max context task on claude-sonnet-4-5-20250929, run:

```
python main.py --task shared_max_context --model anthropic/claude-sonnet-4-5-20250929
```

You can also configure the evaluator model if desired - our public benchmarks use Sonnet 4.5.

### List of Models

A list of avaiable models can be found at our [model library](https://github.com/vals-ai/model-library/blob/main/model_library/config/all_models.json), and also by running `make browse-models` in the model library repository.

To run your own harness or model, just modify the `get_custom_model` function as needed. You will need to implement a function that takes the text input and the documents
and returns a response from the LLM. To see the full documentation on how the SDK works, visit [our docs](https://docs.vals.ai/sdk/running_suites).
