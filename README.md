# Running Corp Fin Benchmark

Our CorpFin benchmark is run via SDK because it requires different parsing of the files depending on the model's context window for one task, which makes it easier to handle everything from code.

There are three main tasks in the benchmark. All have the same data, but they pass the document context to the model in different ways:

- _Exact Pages_: Passes only the exact pages in the document needed to answer the question (generally one to two pages)
- _Shared Max Context_: Passes ~128,000 tokens worth of the document. The selection is guaranteed to have the information needed, and is independent of the model being evaluated.
- _Max Fitting Context_: Passes as much of the document as will fit in the context window of the model, starting from the beginning of the document.

For more details on the benchmark, please refer to the our [public website](https://www.vals.ai/benchmarks/corp_fin_v2).

## Set up

Install the requirements inside requirements.txt. We recommend doing this inside a Python 3.11 [Conda environment.](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

```
pip install -r requirements.txt
```

Make an account on [platform.vals.ai](https://www.platform.vals.ai/auth) with your company email address. Go to the admin page and create a new API key for yourself.

Then run

```
export VALS_API_KEY=<api_key>
```

We recommend adding this to your ~/.bashrc or ~/.zshrc. You will also need to add environment variables for any providers you plan on using (e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.).

Finally, you should add the "Test Suite IDs" to suites.json. These should have generally been provided to you via email, but you can also find them in the platform, by navigating to the "Test Suites" page, clicking the relevant test suite, and looking on the right sidebar under "Test Suite ID".

## Running the benchmark

To run the benchmark, simply do

```
python run_corpfin.py --task shared_max_context --model anthropic/claude-sonnet-4-5-20250929
```

You can also configure the evaluator model if desired - our public benchmarks use Sonnet 4.5.

For the max fitting context window task, we generally add a buffer to our truncation to account for differences in token calculation for models. For example, if a model has a 400,000 token context window, and the buffer is set to 25,000, we will pass in the first 375,000 tokens to the model. You can configure this buffer using `--output-buffer <value>`.

To run your own harness or model, just modify the `get_custom_model` function as needed. You will need to implement a function that takes the text input and the documents
and returns a response from the LLM. To see the full documentation on how the SDK works, visit [our docs](https://docs.vals.ai/sdk/running_suites).
