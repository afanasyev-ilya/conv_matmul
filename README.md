**1. im2col and conv -> matmul conversion**

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

Winograd convolution ideas:

we also use matmul, but save operations for a fixed kernel size. E.g. instead of dowing dot product, we do this:

![Diagram of …](https://miro.medium.com/v2/resize:fit:1082/format:webp/1*Mt1Nqb-dgZ8hsvwqFGc8Xw.png)

and result = [m1 + m2 + m3], [m2 - m3 - m4].

since some of coefficients in m2 and m3 depend only on filter values, we do less multiplications and additions.

**2. 1x1 conv and matmul -> conv conversion**

```
python3 ./matrix.py
```

to check how to do matmul via convolution (1x1)