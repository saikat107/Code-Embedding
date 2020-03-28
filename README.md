# Attention classification Sequence Embedding

This repository provides a prototype of attention based classifier to classify a sequence of token. 

## Steps:

### Data 
Data needs to be saved in 'data/' directory in json file. 

The json object should look like this 

```
data = [
  {
    'code': <sequence of tokens separated by space>,
    'label': integer (0/1)
  },
  ... ... 
]
```

### Procedure

1. Train Word2Vec Model based on the data 
    ```shell script
    python word2vec_train.py  \
        --data_path <List of json files in dataset> \
        --save_model_dirr <Directory where the Word2Vec model will be saved> \
        --model_name <Name of the Word2Vec model to be saves> \
        --epochs <Number of Epochs to train Word2Vec> \
        --embedding_size <Dimension of the Word2Vec word embedding> 
    ```
    For example
    ```shell script
    python word2vec_train.py  \
        --data_path data/demo_train.json data/demo_test.json \
        --save_model_dir wv_models \
        --model_name demo_wv \
        --epochs 100 \
        --embedding_size 256
    ```

2. Run the AttentionEmbedding model.

    There are 3 actions available
    1. `train` :  
        ```shell script
           python attention_main.py \
               --word_to_vec <Word2Vec path> \
               --train_file <Train Json File Path> \
               --model_path <Path where the trained model will be saved> \
               --job train
        ```
        For Example, 
        ```shell script
           python attention_main.py \
               --word_to_vec wv_models/code \
               --train_file data/code_train.json \
               --model_path models/demo_model.bin \
               --job train
        ```
    2. `generate` :
        ```shell script
           python attention_main.py \
             --word_to_vec <Word2Vec path> \
             --test_file <Path of the test json file> \
             --model_path <Trained model path> \
             --test_output_path <Path where the generated test embedding will be saved (Optional)>
        ```
       For example,
       ```shell script
           python attention_main.py \
             --word_to_vec wv_models/code \
             --test_file data/code_test.json \
             --model_path models/demo_model.bin \
             --test_output_path data/test_output.json
        ```
   3. `train_and_generate`: For both Training and Generating. 
