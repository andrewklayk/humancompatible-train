"""Minimal constrained neural network training with 4 optimizers."""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from folktables import ACSDataSource, generate_categories, ACSIncome
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import optimizers
from humancompatible.train.optim import SSG
from humancompatible.train.dual_optim import ALM, PBM, MoreauEnvelope

# Setup
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load folktables data
data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
acs_data = data_source.get_data(states=["VA"], download=True)
definition_df = data_source.get_definitions(download=True)
categories = generate_categories(
    features=ACSIncome.features, definition_df=definition_df
)
df_feat, df_labels, _ = ACSIncome.df_to_pandas(
    acs_data, categories=categories, dummies=True
)

sens_cols = ["SEX_Female", "SEX_Male"]
features = df_feat.drop(columns=sens_cols).to_numpy(dtype="float")
groups = df_feat[sens_cols].to_numpy(dtype="float")
labels = df_labels.to_numpy(dtype="float")

# Split and scale
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    features, labels, groups, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val, groups_train, groups_val = train_test_split(
    X_train, y_train, groups_train, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Convert to tensors and move to device
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
groups_train = torch.tensor(groups_train, dtype=torch.float32).to(device)

# Create dataloader
dataset = TensorDataset(X_train, groups_train, y_train)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

def create_model():
    """Simple neural network."""
    return nn.Sequential(
        nn.Linear(X_train.shape[1], 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    ).to(device)

criterion = nn.BCEWithLogitsLoss()

def get_constraint(model, groups):
    """Constraint: positive rate difference between groups."""
    group_preds = [[] for _ in range(groups.shape[1])]
    for i in range(groups.shape[1]):
        group_preds[i] = (torch.sigmoid(model(X_train)) * groups[:, i].unsqueeze(1)).mean()
    
    # Return max difference in positive rates across groups
    rates = torch.stack(group_preds)
    return (rates.max() - rates.min())

# ============ 1. ADAM (unconstrained) ============
# print("\n=== 1. ADAM ===")
# model = create_model()
# opt = torch.optim.Adam(model.parameters(), lr=0.001)

# for epoch in range(3):
#     losses = []
#     constraints = []
#     for batch_x, batch_groups, batch_y in loader:
#         batch_x, batch_groups, batch_y = batch_x.to(device), batch_groups.to(device), batch_y.to(device)
#         output = model(batch_x)
#         loss = criterion(output, batch_y)
#         loss.backward()
#         opt.step()
#         opt.zero_grad()
#         losses.append(loss.item())
    
#     constraint = get_constraint(model, groups_train).to(device)
#     constraints.append(constraint.item())
#     print(f"Epoch {epoch}: loss={sum(losses)/len(losses):.4f}, constraint={constraint:.4f}")

# # ============ 2. ALM (Augmented Lagrangian) ============
# print("\n=== 2. ALM ===")
# model = create_model()
# opt = MoreauEnvelope(torch.optim.Adam(model.parameters(), lr=0.01))
# dual = ALM(m=1, lr=0.01, momentum=0.5, device=device)
# if hasattr(dual, 'to'):
#     dual = dual.to(device)

# for epoch in range(3):
#     losses = []
#     for batch_x, batch_groups, batch_y in loader:
#         batch_x, batch_groups, batch_y = batch_x.to(device), batch_groups.to(device), batch_y.to(device)
#         output = model(batch_x)
#         loss = criterion(output, batch_y)
#         constraint = get_constraint(model, groups_train).to(device)
        
#         lagrangian = dual.forward_update(loss, constraint.unsqueeze(0))
#         lagrangian.backward()
#         opt.step()
#         opt.zero_grad()
#         losses.append(loss.item())
    
#     constraint = get_constraint(model, groups_train).to(device)
#     print(f"Epoch {epoch}: loss={sum(losses)/len(losses):.4f}, constraint={constraint:.4f}")

# # ============ 3. PBM (Penalty-Barrier Method) ============
# print("\n=== 3. PBM ===")
# model = create_model()
# opt = MoreauEnvelope(torch.optim.Adam(model.parameters(), lr=0.001))
# dual = PBM(m=1, penalty_update='dimin_adapt', gamma=0.7,
#            init_duals=0.001, init_penalties=1., penalty_range=(0.01, 1.),
#            dual_range=(0.01, 100.), device=device)
# if hasattr(dual, 'to'):
#     dual = dual.to(device)

# for epoch in range(3):
#     losses = []
#     for batch_x, batch_groups, batch_y in loader:
#         batch_x, batch_groups, batch_y = batch_x.to(device), batch_groups.to(device), batch_y.to(device)
#         output = model(batch_x)
#         loss = criterion(output, batch_y)
#         constraint = get_constraint(model, groups_train).to(device)
        
#         lagrangian = dual.forward_update(loss, constraint.unsqueeze(0))
#         lagrangian.backward()
#         opt.step()
#         opt.zero_grad()
#         losses.append(loss.item())
    
#     constraint = get_constraint(model, groups_train).to(device)
#     print(f"Epoch {epoch}: loss={sum(losses)/len(losses):.4f}, constraint={constraint:.4f}")

# ============ 4. SSW (Switching Subgradient) ============
print("\n=== 4. SSW ===")
model = create_model()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
opt2 = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    losses = []
    for batch_x, batch_groups, batch_y in loader:
        batch_x, batch_groups, batch_y = batch_x.to(device), batch_groups.to(device), batch_y.to(device)
        output = model(batch_x)
        loss = criterion(output, batch_y)
        constraint = get_constraint(model, groups_train).to(device)

        if constraint.item() > 0:
            constraint.backward()
            opt2.step()
            losses.append(loss.item())
            opt2.zero_grad()
        else:
            loss.backward()
            opt.step()  # No constraint violation, so pass 0 to update
            losses.append(loss.item())
            opt.zero_grad() 

        losses.append(loss.item())
    
    constraint = get_constraint(model, groups_train).to(device)
    print(f"Epoch {epoch}: loss={sum(losses)/len(losses):.4f}, constraint={constraint:.4f}")

print("\n✓ Complete")
