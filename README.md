1. use 

```
python3 ./filter.py --filter EDGES --compute DIRECT_CONV
```

to check how to do convolution via matmul (im2col). Im2col ideas:
https://medium.com/@dmangla3/understanding-winograd-fast-convolution-a75458744ff

Suppose we have input image f of size (4) and filter g of size (3).

f = [1, 2, 3, 4], g = [-1, -2, -3]

Conv does 2 filter multiplications:
1*(-1) + 2*(-2) + 3*(-3)
2*(-1) + 3*(-2) + 4*(-3)

Then, using the im2col technique we convert input image into:
f = [[1, 2, 3], [2, 3, 4]]

and fo matmul f x g (g will be column), which will produce exactly the same result.

NCHW example (instead of 1d): ![Diagram of …](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*PLxQxGGuw0TSfFgE.png)


2. use 

```
python3 ./matrix.py
```

to check how to do matmul via convolution (1x1)