# Running Corp Fin Benchmark

Our CorpFin benchmark is ran via SDK because it requires different parsing of the files depending on the model's context window for one task, which makes it easier to handle everything from code.

For more details on the benchmark, please refer to the our public website where we report the results [vals.ai](https://www.vals.ai/home).

## Running the benchmark

Aside from usual providers' SDKs, you will need to install our SDK using the command `pip install valsai`.

Here are our SDK specific parameters:
- `eval_model`: The model to used as LLM as judge for the evaluation of the model under test's outputs.
- `parallelism`: The number of concurrent calls to the API to get the model's outputs and evaluate them.
- `upload_concurrency`: The number of concurrent calls to upload the results to our platform, so that you can easily monitor, compare and share them online.

Configure the rest of the parameters (temperature, max_tokens, etc.) in the `custom_model_google.py` file.

Then, run the script with:
```bash
python custom_model_google.py
```
