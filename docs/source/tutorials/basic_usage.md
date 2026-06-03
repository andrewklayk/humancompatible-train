---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: hc-dev
  language: python
  name: python3
---

# Basic Usage

This page provides an overview of using humancompatible-train for constrained deep learning on a simple example.

+++

## Idea

+++

The core of the package is formed by Lagrangian-based **dual optimizers**, which are PyTorch Optimizer-like objects that handle the **constrained** part of **constrained deep learning**.

They create, keep track of, and update the **dual parameters** of the constrained minimization problem, as well as calculate the Lagrangian that is then minimized by a standard PyTorch optimizer in place of a loss.

+++

## Simple Example

+++

Let us demonstrate using a **fairness-constrained learning** task, where we want to learn a classifier that is accurate but also satisfies a **demographic parity constraint** - i.e. we would like

$$ | P( Y = 1 | \text{X is Male}) - P ( Y = 1 | \text{X is Female} ) | \leq \epsilon $$

where $ Y $ is the prediction given by our model for sample $ X $, and $ \epsilon $ is some small threshold.

<!-- :::{note}
Note: Normally, this would be an inequality constraint, but for the sake of this example let us handle this case first.
::: -->
---

+++

To enforce demographic parity, we will define a **constraint function** (using the [fairret](https://github.com/aida-ugent/fairret) package) that measures the difference in positive prediction rates between two demographic groups.

The **dual optimizer** will then update the Lagrange multipliers to enforce this constraint during training.

+++

First, let us load and prepare the data. We will use the ACS dataset, containing U.S. Census data, provided by the [folktables](https://github.com/socialfoundations/folktables) package. Feel free to skip this section.

```{code-cell} ipython3
---
tags: [hide-cell]
---
# load data
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from folktables import ACSDataSource, ACSIncome, generate_categories

torch.set_default_dtype(torch.float32)

# load folktables data
data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
acs_data = data_source.get_data(states=["FL"], download=True)
definition_df = data_source.get_definitions(download=True)
categories = generate_categories(
    features=ACSIncome.features, definition_df=definition_df
)
df_feat, df_labels, _ = ACSIncome.df_to_pandas(
    acs_data, categories=categories, dummies=True
)
sens_cols = ["SEX_Female", "SEX_Male"]
features = df_feat.drop(columns=sens_cols).to_numpy(dtype=np.float32)
labels = df_labels.to_numpy(dtype=np.float32)
# one-hot encoding of the sensitive attribute (gender)
groups = df_feat[sens_cols].to_numpy(dtype=np.float32)

# standardize features
scaler = StandardScaler()
features = scaler.fit_transform(features)
# convert to torch tensors
X = torch.tensor(features) ; y = torch.tensor(labels) ; groups = torch.tensor(groups)

dataset_train = torch.utils.data.TensorDataset(X, groups, y)
loader = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True)
criterion = torch.nn.BCEWithLogitsLoss()
```

Initialize the model and optimizer.

```{code-cell} ipython3
from torch.nn import Sequential
from torch.optim import AdamW

def setup_model():

    model = Sequential(
        torch.nn.Linear(features.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )
    model.forward(torch.zeros(features.shape[1])).backward()  # dummy forward/backward pass to construct torch graph for fair comparison
    optimizer = AdamW(model.parameters())
    return model, optimizer
```

Next, we define the **constraint function** for demographic parity, which uses the `fairret.statistic.PositiveRate` class to evaluate positive rates for both groups. \
As a reminder, we expect our constraints to be of the form $ g(...) \leq 0 $ or $ h(...) = 0 $. We want $ g(...) \leq \epsilon $, so we will subtract $ \epsilon $ in the training loop.

```{code-cell} ipython3
from fairret.statistic import PositiveRate

statistic = PositiveRate()

def pr_diff(logit, groups):
    preds = torch.sigmoid(logit)
    stats = PositiveRate()(preds, groups)
    stat_diff = torch.abs(stats[0] - stats[1])
    return stat_diff
```

As a last step, we define our **dual optimizer**. To set it up, we only need to define the **number of constraints** -- in our case, it is 1 -- so it can create the corresponding dual variables, and the **type** of constraint -- equality or inequality. In a following tutorial, we will see how to create *constraint groups* with different types and hyperparameters.

```{code-cell} ipython3
from humancompatible.train.dual_optim import ALM

dual_optimizer = ALM(m=1, lr=0.01, is_ineq=True)
```

Finally, we write our training loop. In addition to the forward pass and loss calculation, we add a constraint calculation step (0.05 is our $ \epsilon $ threshold).

Then, the `forward_update` step does two things:

- Updates the dual variables based on the constraint violation,
- Calculates the Lagrangian based on loss and constraint violation.

We then perform a backward pass on the Lagrangian and minimize it using a normal PyTorch optimizer.

```{code-cell} ipython3
model, optimizer = setup_model()
epochs = 10
```

```{code-cell} ipython3
for epoch in range(epochs):
    # eval
    model.eval()
    logit = model(X)
    train_loss = criterion(logit, y).item()
    train_fair = pr_diff(logit, groups).item()
    print(f"Epoch: {epoch}, loss: {train_loss}, constraint: {train_fair}")
    
    # train
    model.train()
    for batch_feat, batch_groups, batch_label in loader:
        optimizer.zero_grad()
        logit = model(batch_feat)
        loss = criterion(logit, batch_label)
        
        constraint = pr_diff(logit, batch_groups) - 0.05
        lagr = dual_optimizer.forward_update(loss, constraint.unsqueeze(0))
        lagr.backward()
    
        optimizer.step()
```

We obtain a respectable loss value, while keeping the fairness violation below the threshold!

+++

Just in case, let's check what happens if we train the model without constraints:

```{code-cell} ipython3
model, optimizer = setup_model()
```

```{code-cell} ipython3
for epoch in range(epochs):
    # eval
    model.eval()
    logit = model(X)
    train_loss = criterion(logit, y).item()
    train_fair = pr_diff(logit, groups).item()
    print(f"Epoch: {epoch}, loss: {train_loss}, constraint: {train_fair}")
    
    # train
    model.train()
    for batch_feat, batch_groups, batch_label in loader:
        optimizer.zero_grad()
        logit = model(batch_feat)
        loss = criterion(logit, batch_label)

        loss.backward()
        optimizer.step()
```

The absolute difference in positive rates is two times higher than what we wanted!

+++

Further reading:

```{code-cell} ipython3

```
