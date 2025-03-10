
## Package
```bash
pip install torch torch-geometric matplotlib scikit-learn numpy

```

## Run MUTAG example
```bash
python Classification.py --dataset MUTAG --num_heads 4 --kernels WL SP RW GL --GL_k 5 --num-layers 3 --hop 2 --outdir test

```
