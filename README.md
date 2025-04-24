# Running Corp Fin Benchmark

Our CorpFin benchmark is run via SDK because it requires different parsing of the files depending on the model's context window for one task, which makes it easier to handle everything from code.

For more details on the benchmark, please refer to the our public website where we report the results [vals.ai](https://www.vals.ai/home).

## Set up

Aside from usual providers' SDKs, you will need to install our SDK using the command `pip install valsai`.

Make an account on [platform.vals.ai](https://www.platform.vals.ai/auth) with your company email address. Go to the admin page and create a new API key for yourself. Make an environment variable for `VALS_API_KEY` with this key.

## Running the benchmark

Here are our SDK specific parameters:
- `eval_model`: The model to used as LLM as judge for the evaluation of the model under test's outputs.
- `parallelism`: The number of concurrent calls to the API to get the model's outputs and evaluate them.
- `upload_concurrency`: The number of concurrent calls to upload the results to our platform, so that you can easily monitor, compare and share them online.

Configure the rest of the parameters (temperature, max_tokens, etc.) in the `custom_model_corp_fin_dataset.py` file.

Then, run the script with:
```bash
python custom_model_corp_fin_dataset.py
```

You can also run any custom model on the datasets with the run_example.py file included in this repository.

See our [examples](https://github.com/vals-ai/vals-sdk/tree/main/examples) folder for more guidance on using the SDK.
