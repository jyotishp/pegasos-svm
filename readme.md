# PEGASOS SVM

## Setting up

- Run `pip install -r requirements.txt` to install required packages.
- Download and extract the MNIST fashion dataset files to `data` directory.

## Running the code

The working directory should be `src`.

### Without kernel

```
python svm.py --dataset_dir ../data --iterations 10000
```

### With kernel
```
python svm.py --dataset_dir ../data --iterations 2 --kernel
```

## Bonus

### Without kernel

```
python svm-multiclass.py --dataset_dir ../data --iterations 10000
```

### With kernel
```
python svm-multiclass.py --dataset_dir ../data --iterations 2 --kernel
```