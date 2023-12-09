# Horizontal Federated Learning Example

This is an example of federated learning (FL) using the Singa framework. In FL, there is a server and a set of clients. Each client has a local dataset. 
In each iteration, each client trains the model using its local dataset and uploads the model gradient to the server, which aggregates to get the global
gradient using the Federated Average algorithm. The server sends the global gradient to all clients for iterative model training. 
This example uses the Bank dataset and an MLP model in FL.

## Preparation

Go to the Conda environment that contains the Singa library, and run

```bash
pip install -r requirements.txt
```

Download the bank dataset and split it into 3 partitions.

```bash
# 1. download the data from https://archive.ics.uci.edu/ml/datasets/bank+marketing
# 2. put it under the /data folder
# 3. run the following command which:
#    (1) splits the dataset into N subsets
#    (2) splits each subsets into train set and test set (8:2)
python -m bank N
```

## Run the example

Run the server first (set the number of epochs to 3)

```bash
python -m src.server -m 3 --num_clients 3
```

Then, start 3 clients in different terminal

```bash
python -m src.client --model mlp --data bank -m 3 -i 0 -d non-iid
python -m src.client --model mlp --data bank -m 3 -i 1 -d non-iid
python -m src.client --model mlp --data bank -m 3 -i 2 -d non-iid
```

Finally, the server and clients finish the FL training. 