# Dynamic learned indexes

Collection of experiments on the prototype of learned dynamicity for high-dimensional data


## How to use:
### With Docker
1. Create the docker image
```
docker build -t dynamic-indexing -f Dockerfile . --network host
```
2. Run the docker image
```
docker run -it -p 8888:8888 -v <<current-path-full>>\\outputs:/learned-dynamicity/outputs dynamic-indexing /bin/bash
```
3. Explore the notebooks:
```
jupyter-lab --no-browser --ip "*" --NotebookApp.autoreload=True --NotebookApp.password=''
```
4. Run the experiments:
```
cd scripts
# will produce a new folder in 'experiments/'
python run-experiments.py ../config.yml 
```
### Without Docker:
0. Prerequisites:
- Python3.6 and higher
- Rust (not necessary if running only on the SIFT dataset)

1. Create a virtual environment, activate it and pre-install the thrid-party libraries and the source package:
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install --editable .
```
2. Download the datasets:
```
python scripts/download-datasets.py
```
3. Explore the notebooks:
```
jupyter-lab --no-browser --ip "*" --NotebookApp.autoreload=True --NotebookApp.password=''
```
4. Run the experiments:
```
cd scripts
# will produce a new folder in 'experiments/'
python run-experiments.py ../config.yml 
```

Additionally, you can explore the Figures in 'figures/' and experiment files in 'experiments/' 

=======
## Initial concept
Dynamic LMI is a dynamically reshapable structure based on the amount of incoming data and their distribution. The system composed of 3 parts -- dynamic **operations** (lowest layer of abstraction), **database** deciding which operation to call, and a specific **use cases** (e.g.: "insert 5k objects").

### Structure constraints:
#### Leaf nodes:
- min, max leaf node capacity (how many objects can be contained within 1 leaf node)
- min, max number of children (how many children nodes be contained within 1 inner node)

### Operations:
- `insert(objects)`
    - the most fundamental operation of data insertion
    - traverses the tree to find a leaf node with the highest probability and inserts data into the leaf node
    - does not manipulate with the structure or trigger any other operation.
    - based on newly inserted data, a `inconsistencies` data structure gets updated if any violation of `constrains` was detected

- `deepen(n)`
    - transforms a leaf node to an inner node, creates `n` new child leaf nodes
    - applicable to leaf nodes only
- `broaden(nodes, n)`
    - creates `n` new sibling nodes, removes `nodes` and redistributes their objects into the new nodes
- `retrain(inner_node, n)`
    - creates a new model to represent `inner_node`, creates new `n` children
    
### Orchestrator (Databse):
Its primary purpose is to react to inconsistencies detected after `insert` by triggering an appropriate **operation(s)**
