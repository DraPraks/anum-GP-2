A SETAR model is a type of autoregressive model that switches between 2 thresholds depending on past data. In this project, the switch is yesterday's return:
    
- If $R_{t-1} >= 0$ --> use the bullish equation
- If $R_{t-1} < 0$ --> use the bearish equation

### Goal
Fit some set of parameters so that, given the last 2 daily returns, you can predict today's return, then measure how well that prediction holds on unseen days. 

The data given are trading day closing prices. We calculate returns simply:

$R_t = (P_t -P_{t-1})/P_{t-1}$

The parameters we are trying to find are the intercept and 2 lag coefficients in each regime. The problem is **trying to obtain these coefficients from an overdetermined system**.

### Regression Model
Regression is a rule that predicts some number $y$ from known numbers $x_1, ...$ . 

$y = \beta_0 + \beta_1x_1 + ... + \beta_px_p + \epsilon$

The $\beta$s are unknown. The $x$s and $y$ are observed. We use least squares to choose $\beta$.

### Least Squares
Write one observation as a row. Stack every observation and the model is $A\vec{x} \approx \vec{b}$. 

Row $i$ says “the dot product of this row with $\vec{x}$ should equal $b_i$.” The leftover on that row is the residual

$$r_i = (A\vec{x})_i - b_i.$$

If there are more rows than unknowns, no $\vec{x}$ makes every $r_i$ zero. Least squares picks the $\vec{x}$ that makes the vector of residuals as short as possible in the Euclidean norm:

$$\vec{x}_{\mathrm{LS}} = \arg\min_x \|A\vec{x} - \vec{b}\|_2 = \arg\min_x \sum_i r_i^2.$$

“Least squares” means that: minimize the sum of squared errors. Squaring does two things. It treats a miss of $+0.02$ and a miss of $-0.02$ as the same, and it punishes one large miss more than several small ones. 

$A\vec{x}$ is a linear combination of the columns of $A$. The best $\vec{x}$ is the one whose combination lands closest to $\vec{b}$. The residual $\vec{b}-A\vec{x}$ is then perpendicular to every column of $A$:

$A^\top (A\vec{x} - \vec{b}) = 0.$


That orthogonality condition is the normal equation $A^\top A\,\vec{x} = A^\top\vec{b}$. QR solves the same minimization without forming $A^\top A$. Both methods below are algorithms for this one $\vec{x}_{\mathrm{LS}}$. Neither is a training loop. $A$ and $\vec{b}$ are known; the solver returns the minimizer.

### What autoregression is
Autoregression is regression in which the predictors are earlier values of the same series. An AR(2) is

$$R_t = \alpha + \phi_1 R_{t-1} + \phi_2 R_{t-2} + \varepsilon_t.$$

“Auto” means the series is explained by its own lags. It is still a linear model in the coefficients. The difference is what sits on the right-hand side: lags of $R$, not separate variables such as volume or interest rates.

A Self-Exciting Threshold Autoregressive (SETAR) uses a lag of the series itself as that threshold variable. The regime on day $t$ is read off $R_{t-1}$.

Here $p=2$, delay $d=1$, cut $c=0$:

$R_t =
\begin{cases}
\alpha_1 + \phi_{1,1} R_{t-1} + \phi_{1,2} R_{t-2} + \varepsilon_t & R_{t-1} \ge 0 \\
\alpha_2 + \phi_{2,1} R_{t-1} + \phi_{2,2} R_{t-2} + \varepsilon_t & R_{t-1} < 0.
\end{cases}$

### From the model to $A\vec{x}=\vec{b}$

Write the indicator:

$$I_t = \mathbf{1}(R_{t-1}\ge 0)$$. 

Then one equation covers both regimes:

$R_t = I_t(\alpha_1 + \phi_{1,1} R_{t-1} + \phi_{1,2} R_{t-2}) + (1-I_t)(\alpha_2 + \phi_{2,1} R_{t-1} + \phi_{2,2} R_{t-2}) + \varepsilon_t.$

Stack every day that has two lags ($t \ge 3$) and the unknown vector is

$$\vec{x} = [\alpha_1,\ \phi_{1,1},\ \phi_{1,2},\ \alpha_2,\ \phi_{2,1},\ \phi_{2,2}]^\top.$$

What we know are the day to day changes which multiplies $$\vec{x}$$. We compile this known data in a matrix $A$, where each row $t$ of $A$ is the known multipliers of those six numbers, and $b_t = R_t$:

- bullish: $[1,\ R_{t-1},\ R_{t-2},\ 0,\ 0,\ 0]$
- bearish: $[0,\ 0,\ 0,\ 1,\ R_{t-1},\ R_{t-2}]$

There are hundreds of rows and six columns, so $A\vec{x}=\vec{b}$ has no exact solution. Least squares is the criterion that picks $\vec{x}$ anyway.

Method 1 - normal equations and LU

Set the derivative of $\|A\vec{x}-\vec{b}\|_2^2$ to zero:

$\frac{\partial}{\partial \vec{x}} \|A\vec{x}-\vec{b}\|_2^2
= 2A^{\top}(A\vec{x}-\vec{b})
= 0$

The minimizer satisfies the square system:

$$G\vec{x}=\vec{c},\qquad G=A^\top A,\qquad \vec{c}=A^\top\vec{b}.$$

Because a row lives in only one block of columns, $G$ is block diagonal: one $3\times 3$ Gram matrix for bullish days and one for bearish days. The code still forms the full $6\times 6$ and factors it.

```
NORMAL_EQUATIONS(A, b)
    G <- Aᵀ A
    c <- Aᵀ b
    (L, U, P) <- LU_PARTIAL_PIVOT(G)
    y <- FORWARD_SUBSTITUTE(L, P b)
    x <- BACK_SUBSTITUTE(U, y)
    return x
LU_PARTIAL_PIVOT(G) // G is n×n, here n = 6
    U <- copy(G)
    L <- I
    P <- I
    for k = 0 .. n-1
        i* <- argmax_{i ≥ k} |U[i, k]|
        if |U[i*, k]| ≈ 0: raise singular
        swap rows k and i* of U and of P
        if k > 0: swap L[k, 0:k) with L[i*, 0:k)
        for i = k+1 .. n-1
            m <- U[i, k] / U[k, k]
            L[i, k] <- m
            U[i, k:n) <- U[i, k:n) − m · U[k, k:n)
    return (L, U, P)
```


Forward substitution solves $Ly = Pb$ top to bottom ($L$ has 1s on the diagonal). Back substitution solves $Ux = y$ bottom to top.
This path is cheap on a $6\times 6$ matrix. It is also the less stable path: $\kappa_2(G) \approx \kappa_2(A)^2$.

Method 2 - Givens QR
TODO