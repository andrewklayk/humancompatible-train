Learning Directed Acyclic Graphs (DAGs) from Data
==================================================

Overview
--------

This example demonstrates how to learn a **Directed Acyclic Graph (DAG)** from data using constrained optimization. We follow an approach inspired by the `Cooper <https://cooper.readthedocs.io/>`_ library's DAG learning example.

In this example, we:

1. Generate synthetic data from a linear structural equation model
2. Define a constrained optimization problem to recover the underlying graph
3. Use the Augmented Lagrangian Method (ALM) to solve the problem
4. Visualize both the learned and ground truth graphs

What is a DAG?
--------------

A Directed Acyclic Graph (DAG) is a graph where:

- Nodes represent variables or features
- Directed edges represent causal relationships
- There are no cycles (acyclic property)
- The acyclic property ensures a topological ordering exists

DAG learning is useful in causal inference, discovering variable dependencies, and understanding structural relationships in data.

Data Generation
---------------

We start by generating synthetic data from a linear structural equation model with Gaussian noise:

.. code-block:: python

    import torch
    import numpy as np
    import math
    
    def generate_data(n, d, n_causes, noise_std, device):
        """Generate data from a linear structural equation model with Gaussian noise.
        
        Args:
            n: number of samples
            d: number of features
            n_causes: number of root nodes (nodes with no parents)
            noise_std: standard deviation of the noise
            device: torch.device
        
        Returns:
            X: Data matrix of shape (n, d)
            A: Adjacency matrix of shape (d, d)
        """
        # Generate adjacency matrix
        A = torch.zeros(d, d, device=device)
        
        for i in range(n_causes, d):
            # Each node (except roots) has random parents from previous nodes
            parents = 0 if i == 1 else torch.randperm(i)[:np.random.randint(1, i)]
            A[i, parents] = 1
        
        # Verify acyclic property
        assert torch.trace(torch.linalg.matrix_exp(A)).item() == d, "A is not a DAG"
        
        # Generate data: X_i = sum(X_parents_i) + noise_i
        noise = noise_std * torch.randn(n, d, device=device)
        X = torch.zeros(n, d, device=device)
        
        for i in range(d):
            parents = torch.nonzero(A[i]).flatten()
            X[:, i] = X[:, parents].sum(dim=1) + noise[:, i]
        
        # Improve conditioning
        X /= math.sqrt(d)
        
        return X, A

**Parameters:**

- ``n``: Number of samples (5,000 in this example)
- ``d``: Number of features/nodes (8 in this example)
- ``n_causes``: Number of root nodes with no parents (2 in this example)
- ``noise_std``: Standard deviation of Gaussian noise (0.01 in this example)

Training Setup
--------------

We formulate the DAG learning problem as a constrained optimization problem:

.. math::

    \min_{A \in \{0, 1\}^{d \times d}} \left\| X - XA \right\|_F^2
    
    \text{subject to:} \quad \text{tr}(e^A) = d

The constraint ensures the adjacency matrix ``A`` represents a valid DAG:

- The exponential matrix ``exp(A)`` has trace equal to ``d`` if and only if ``A`` is acyclic
- This is an algebraic constraint that replaces the combinatorial acyclicity check

**Implementation:**

.. code-block:: python

    from humancompatible.train.dual_optim import ALM
    from torch.optim import AdamW
    
    # Initialize adjacency matrix as a learnable parameter
    A = torch.nn.Parameter(torch.randn(D, D, device=DEVICE) / math.sqrt(D))
    
    # Optimizer for the primal variable (adjacency matrix)
    optimizer = AdamW(params=[A], lr=PRIMAL_LR)
    
    # Dual optimizer using Augmented Lagrangian Method
    dual_opt = ALM(m=1)  # m=1 constraint
    
    # Constraint function
    constraint = lambda A: torch.trace(torch.linalg.matrix_exp(A)) - d

Training Loop
-------------

The training procedure alternates between:

1. **Primal step**: Update ``A`` to minimize the Lagrangian
2. **Dual step**: Update Lagrange multipliers to enforce constraint satisfaction

.. code-block:: python

    for i in range(N_STEPS):
        # Project to valid range [0, 1] and remove diagonal
        A.data.fill_diagonal_(0)
        A.data.clamp_(min=0, max=1.0)
        
        # Compute loss: reconstruction error
        loss = torch.square(torch.linalg.norm(X - X @ A.T, ord="fro"))
        
        # Compute constraint violation
        cviol = constraint(A)
        
        # Update Lagrangian
        lagrangian = dual_opt.forward_update(loss, cviol.unsqueeze(0))
        
        # Gradient descent on primal variable
        lagrangian.backward()
        optimizer.step()
        optimizer.zero_grad()

**Key steps:**

- **Diagonal removal**: No self-loops allowed (``A.fill_diagonal_(0)``)
- **Value clamping**: Adjacency values are bounded to [0, 1] (``A.clamp_(min=0, max=1.0)``)
- **Loss computation**: Measures how well ``A`` predicts the data
- **Constraint enforcement**: The ALM solver tracks dual variables to enforce the acyclicity constraint

Results and Visualization
--------------------------

After training, we can visualize the learned adjacency matrix alongside the ground truth:

.. code-block:: python

    import networkx as nx
    import seaborn as sns
    from matplotlib import pyplot as plt
    
    # Create network graph
    G = nx.DiGraph()
    G.add_nodes_from(range(D))
    
    for i in range(D):
        for j in range(D):
            if A[i, j] != 0:
                G.add_edge(j, i)
    
    # Visualize
    pos = nx.shell_layout(G)
    plt.figure(figsize=(5, 2))
    nx.draw(G, pos, with_labels=True, font_weight="bold")
    plt.show()

**Visualization outputs:**

1. **Adjacency Heatmaps**: Compare learned, ground truth, and difference matrices
2. **Training Progress**: Track loss, constraint violation, and dual parameters over iterations

The quality of recovery depends on:

- **Dataset size**: Larger datasets improve recovery
- **Noise level**: Lower noise enables better recovery
- **Training iterations**: More iterations improve convergence
- **Graph density**: Sparser graphs are easier to recover

Applications
-----------

DAG learning is useful for:

- **Causal discovery**: Inferring causal relationships from observational data
- **Biological networks**: Discovering gene regulatory networks
- **Financial systems**: Understanding dependencies between economic indicators
- **Knowledge graphs**: Learning structured relationships from data
- **Feature importance**: Understanding variable interactions

See Also
--------

- :doc:`api_reference` for the ALM solver and optimization utilities
- `Cooper Documentation <https://cooper.readthedocs.io/>`_ for more constrained optimization examples
- The full notebook: ``examples/learn_DAG.ipynb``
