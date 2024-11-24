# Instructions

## Step 1: Prepare Data

- Open the `/data/scripts/select_sentences.sh`
at `line 8` and `line 9` you can specify the path for the data (which in our case are the `combined-eng.txt-0` and `combined-log.txt-0`).
The `-0` means that the data have been preprocessed.

- run the command: `./select_sentences <number_of_suffled_lines>`
if no `<number_of_suffled_lines>` is specifiied it will use the whole dataset.
If `<number_of_suffled_lines>` exist as arguments, it will create a training data file with the specified number of sentences.

- In both cases, the script is going to create the `testing` and the `validating` data, where the validating data will be 10% of the testing data.

## Step 2: Train the model

- Change the paths in the `batch_train.sh` file, and the resources if you want.

- Open the `train.py` and change the paths, so that they will point to your new create data from `Step 1`.

``` txt
# Paths to your data
input_file = 'data/full_12m_sentences/input_sentences.txt'
output_file = 'data/full_12m_sentences/output_logical.txt'
tokenized_output_file = 'data/full_12m_sentences/tokenized_data.json'
```

For the `tokenized_output_file`, you just need to specify the name of the file and the location that you want it to be saved. It should be empty, if you run the program for the first time.
If you have run the program again in the past for **the same dataset** you can comment out the
``` python
tokenize_data(input_file, output_file, tokenized_output_file)
```
to reduce time.


## Step 3: Validate data

### Validate one model.

- Change again the `log paths` in the `evaluate_model_job.sh`.
- Open the `evaluate_model.py` and:
1. Change the path of the model you want to test.
``` python
    model = load_model('data/500k_sentences_suffled/t5_model')
```
2. Change the path for the validation data at lines **53** and **58**.

``` python
    with open('data/500k_sentences_suffled/input_sentences_500k_val.txt', 'r') as f:
    ....
    with open('data/500k_sentences_suffled/output_logical_500k_val.txt', 'r') as f:
```

- Run the `./evaluate_model_job.sh`
It will create a new file called `predictions_and_references.txt`, and it will save each prediction and its true value for every sentence that the model will be validated on, and in the end it will print the percentage of the prediction that are **exactly** the same with the true values.

### Testing and Comparing multiple models on Custom Data.

- Open the file `multiple_models_reference.py`.
Change the paths in the array at line **24**, with a value for every model you want to test.

``` python
    model_paths_names = [
      ('data/full_12m_sentences/t5_model_3_epochs','12m Sentences'),
      ('data/500k_sentences_suffled/t5_model','500k SUFFLED Sentences'),
      ('data/500k_last_sentences/t5_model','500k Last Sentences')
    ]  # Add your actual model paths here
```

- write the `sentence`  you want to test in the `input.txt` file.

- Run the python file
`python3 multiple_models_reference.py`
A new `output_multiple_models.txt` will be created with the predictions for each model, for easy comparison.
Example:

``` text
Input: OrangeRed is an instance of Orange

Model: 12m Sentences
Output: ( instance OrangeRed Orange )

Model: 500k SUFFLED Sentences
Output: ( instance OrangeRed Orange )

Model: 500k Last Sentences
Output: ( exists (? H? P? DO? IO ) ( and ( instance? H Human ) ( names "OrangeRed"? H ) ( instance? P Ingesting ) ( experiencer? P? H ) ( attribute? DO Female ) ( names "OrangeRed"? DO ) ( instance? DO Human ) ( objectTransferred? P? DO ) ) )

==================================================
```





