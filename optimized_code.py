! pip install torch deap scikit-learn pandas numpy shap
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from deap import base, creator, tools, algorithms
import random
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# CONFIG / DEVICE
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "Fire_dataset_cleaned.csv"
RANDOM_SEED = 42
POPULATION_SIZE = 6
N_GENERATIONS = 5
GA_CXPB = 0.5
GA_MUTPB = 0.3
FINAL_EPOCHS = 15  # train final model longer
GA_EVAL_EPOCHS = 3  # quick train during GA evaluation

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# -------------------------
# 1) Load & prepare data
# -------------------------
df = pd.read_csv(DATA_PATH)

# If dataset has one-hot encoded target "Classes_not fire" like in your code, use it.
# Otherwise adjust to your dataset's naming.
if "Classes_not fire" in df.columns:
    # original: Classes_not fire == 1 -> not fire; we want 1 -> fire
    # map so target = 1 for 'fire' and 0 for 'not fire'
    y = df["Classes_not fire"].apply(lambda x: 0 if x else 1).astype(int)
    X_df = df.drop(columns=["Classes_not fire"]).copy()
elif "Classes" in df.columns:
    # accept 'fire'/'not fire' strings
    if df["Classes"].dtype == object:
        y = df["Classes"].map({'fire': 1, 'not fire': 0}).astype(int)
    else:
        y = df["Classes"].astype(int)
    X_df = df.drop(columns=["Classes"]).copy()
else:
    raise ValueError("Could not locate target column. Please provide either 'Classes' or 'Classes_not fire'.")

# Drop obviously extraneous columns if present
for c in ['Unnamed: 0', 'Region', 'day', 'month', 'year']:
    if c in X_df.columns:
        X_df = X_df.drop(columns=[c])

# Keep original column names and indices
orig_columns = list(X_df.columns)

# Scale features and keep scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

# -------------------------
# 2) XGBoost feature importance -> pick top k features
# -------------------------
k = min(10, X_scaled.shape[1])
xgb = XGBClassifier(n_estimators=100, random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_scaled, y)
fi = xgb.feature_importances_
top_indices = np.argsort(fi)[-k:]
# preserve original order among selected features
top_indices = sorted(top_indices.tolist())
top_features = [orig_columns[i] for i in top_indices]

# create selected arrays
X_top = X_scaled[:, top_indices]

# train/val split on selected features
X_train, X_val, y_train, y_val = train_test_split(X_top, y.values, test_size=0.2, random_state=RANDOM_SEED)

print("\n✅ Top features (used for input prompts):")
for i, f in enumerate(top_features, 1):
    print(f" {i}. {f}")

# -------------------------
# 3) Safer Transformer model
#    Input -> Linear(embed->hidden_dim) -> Transformer(d_model=hidden_dim) -> Classifier
# -------------------------
class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2, num_classes=2):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2, dim_feedforward=hidden_dim*2,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [batch, features]
        x = self.fc_in(x)           # [batch, hidden]
        x = x.unsqueeze(1)          # [batch, seq=1, hidden]
        x = self.transformer(x)     # [batch, seq=1, hidden]
        x = x.mean(dim=1)           # [batch, hidden]
        x = self.dropout(x)
        return self.out(x)          # logits

# -------------------------
# 4) Utility: clip GA individual to valid ranges
# -------------------------
def clip_individual(ind):
    # ind layout: [lr, dropout, hidden_dim, num_layers, batch_size]
    # lr
    ind[0] = float(max(1e-6, min(ind[0], 1e-1)))   # keep lr in [1e-6, 1e-1]
    # dropout
    ind[1] = float(max(0.0, min(ind[1], 0.9)))     # [0, 0.9] safe
    # hidden dim -> choose from allowed set
    allowed_hidden = [32, 64, 128]
    # if mutated to non-int, round to nearest allowed
    try:
        h = int(round(ind[2]))
    except Exception:
        h = allowed_hidden[0]
    # pick nearest
    ind[2] = min(allowed_hidden, key=lambda a: abs(a - h))
    # num_layers
    ind[3] = int(max(1, min(int(round(ind[3])), 4)))
    # batch size -> from allowed choices
    allowed_batch = [8, 16, 32, 64]
    try:
        b = int(round(ind[4]))
    except Exception:
        b = allowed_batch[1]
    ind[4] = min(allowed_batch, key=lambda a: abs(a - b))
    return ind

# -------------------------
# 5) GA: create individual / population and operators
# -------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def create_individual():
    lr = 10 ** random.uniform(-6, -3)          # initial lr in [1e-6, 1e-3]
    dropout = random.uniform(0.05, 0.4)        # initial dropout [0.05, 0.4]
    hidden = random.choice([32, 64, 128])
    layers = random.choice([1, 2, 3])
    batch = random.choice([8, 16, 32, 64])
    ind = [lr, dropout, hidden, layers, batch]
    return creator.Individual(clip_individual(ind)) # Ensure Individual type is created

toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# custom mutation: gaussian on numeric genes, but clip afterwards
def custom_mutate(individual, mu=0, sigma=0.2, indpb=0.2):
    for i in range(len(individual)):
        if random.random() < indpb:
            if i in (0,1):  # continuous (lr, dropout)
                individual[i] = individual[i] + random.gauss(mu, sigma * (abs(individual[i]) + 1e-6))
            else:
                # integer genes: add gaussian then round
                individual[i] = individual[i] + int(round(random.gauss(mu, sigma)))
    # clip to valid ranges after mutation
    clip_individual(individual) # Clip in place
    return individual,

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", custom_mutate, mu=0, sigma=0.2, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)

# -------------------------
# 6) Fitness evaluation: train small model for a few epochs and return validation accuracy
# -------------------------
def evaluate_model(individual):
    # ensure individual valid - pass a copy to clip_individual
    ind = clip_individual(list(individual))
    lr, dropout, hidden_dim, num_layers, batch_size = ind
    lr = float(lr); dropout = float(dropout); hidden_dim = int(hidden_dim)
    num_layers = int(num_layers); batch_size = int(batch_size)


    # create model
    model = TemporalTransformer(input_dim=X_train.shape[1],
                                hidden_dim=hidden_dim,
                                num_layers=num_layers,
                                dropout=dropout,
                                num_classes=2).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # dataloaders
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    Ytr = torch.tensor(y_train, dtype=torch.long)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    Yv = torch.tensor(y_val, dtype=torch.long)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, Ytr),
                                               batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xv, Yv),
                                             batch_size=batch_size, shuffle=False)

    # quick training loop
    model.train()
    for epoch in range(GA_EVAL_EPOCHS):
        for xb, yb in train_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    # evaluate
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            logits = model(xb)
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    acc = correct / total if total > 0 else 0.0

    # store last prediction summary on validation set (for printing in loop)
    evaluate_model.last_pred_acc = acc
    # not storing label here (we want final model to predict on user input)
    return (acc,)

toolbox.register("evaluate", evaluate_model)

# -------------------------
# 7) Interactive user input (asked BEFORE GA as requested)
#    Inputs must match top_features order printed earlier
# -------------------------
print("\n--- Enter values for each feature (numeric). Press Enter after each. ---")
user_vals = []
for feat in top_features:
    while True:
        raw = input(f"{feat}: ").strip()
        try:
            val = float(raw)
            user_vals.append(val)
            break
        except ValueError:
            print("  Invalid input — please enter a numeric value (e.g., 25 or 25.0).")

user_arr = np.array(user_vals).reshape(1, -1)
# scale user input using the original scaler: need to build a full-row with original columns then select indices
# Create a zero-row for all original columns and insert the provided top feature values
full_row = np.zeros((1, len(orig_columns)))
for i, idx in enumerate(top_indices):
    full_row[0, idx] = user_vals[i]
full_row_scaled = scaler.transform(full_row)  # scale full feature vector
user_top_scaled = full_row_scaled[:, top_indices]  # select only top features
user_tensor = torch.tensor(user_top_scaled, dtype=torch.float32).to(DEVICE)

# -------------------------
# 8) Run GA (print progress each generation and prediction status using a small model built from best individual)
# -------------------------
pop = toolbox.population(n=POPULATION_SIZE)

print("\n🚀 Starting GA Optimization for CLASSIFICATION task...")
print("=========================================================\n")

for gen in range(1, N_GENERATIONS + 1):
    # variation (crossover + mutation)
    offspring = algorithms.varAnd(pop, toolbox, cxpb=GA_CXPB, mutpb=GA_MUTPB)

    # clip offspring individuals to valid ranges (important after crossover/mutation)
    # No need to explicitly clip here, custom_mutate handles clipping

    # evaluate fitness
    fits = list(map(toolbox.evaluate, offspring))
    for ind, fit in zip(offspring, fits):
        ind.fitness.values = fit

    # select next population
    pop = toolbox.select(offspring, k=len(pop))

    # report best this generation
    best = tools.selBest(pop, 1)[0]
    best_acc = best.fitness.values[0]
    # Use evaluate_model.last_pred_acc as approximation to show validation behaviour
    gen_pred_acc = getattr(evaluate_model, 'last_pred_acc', None)

    # Build a quick model from best (short train) to also show prediction on user input (so user sees Fire/No Fire per generation)
    # We will train a quick model using best individual (light training)
    b_lr, b_drop, b_hidden, b_layers, b_batch = clip_individual(list(best)) # Use clip_individual here to ensure valid params for the quick model
    quick_model = TemporalTransformer(input_dim=X_train.shape[1],
                                      hidden_dim=int(b_hidden),
                                      num_layers=int(b_layers),
                                      dropout=float(b_drop),
                                      num_classes=2).to(DEVICE)
    # train quick model briefly on training data
    opt = optim.Adam(quick_model.parameters(), lr=float(b_lr))
    crit = nn.CrossEntropyLoss()
    quick_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_train,dtype=torch.float32),
                                                                              torch.tensor(y_train,dtype=torch.long)),
                                              batch_size=int(b_batch), shuffle=True)
    quick_model.train()
    for _ in range(2):
        for xb, yb in quick_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(quick_model(xb), yb)
            loss.backward(); opt.step()

    quick_model.eval()
    with torch.no_grad():
        out = quick_model(user_tensor)
        probs = torch.softmax(out, dim=1).cpu().numpy().flatten()
        gen_pred_label = int(np.argmax(probs))

    # Print generation summary
    print(f"🌀 Generation {gen}/{N_GENERATIONS}")
    print("---------------------------------------------------------")
    print(f"⭐ Best Fitness: {best_acc:.4f}")
    print(f"   Hyperparams: LR={b_lr:.6f}, Dropout={b_drop:.3f}, Hidden={int(b_hidden)}, Layers={int(b_layers)}, Batch={int(b_batch)}")
    print(f"   Selected Features: {len(top_features)}/{k}")
    if gen_pred_label == 1:
        print(f"   🔥 Prediction (on your input): FIRE (prob={probs[1]:.3f})")
    else:
        print(f"   🌿 Prediction (on your input): NO FIRE (prob={probs[0]:.3f})")
    print("")

# final best individual
best = tools.selBest(pop, 1)[0]
# No need to convert to list here before accessing fitness
print("🏁 GA Optimization Complete!")
print(f"🏆 Best Fitness Score: {best.fitness.values[0]:.4f}")
print(f"📊 Selected Features: {len(top_features)}/{k}")
best_params = clip_individual(list(best)) # Use clip_individual here to ensure valid params for printing
print(f"🏅 Best individual: LR={best_params[0]:.6f}, Dropout={best_params[1]:.3f}, Hidden={int(best_params[2])}, Layers={int(best_params[3])}, Batch={int(best_params[4])}")

# -------------------------
# 9) Train final model with best hyperparams (longer), then predict on user input
# -------------------------
final_lr, final_drop, final_hidden, final_layers, final_batch = best_params # Use best_params here
final_model = TemporalTransformer(input_dim=X_train.shape[1],
                                  hidden_dim=int(final_hidden),
                                  num_layers=int(final_layers),
                                  dropout=float(final_drop),
                                  num_classes=2).to(DEVICE)

optimizer = optim.Adam(final_model.parameters(), lr=float(final_lr))
criterion = nn.CrossEntropyLoss()

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_train,dtype=torch.float32),
                                                                         torch.tensor(y_train,dtype=torch.long)),
                                           batch_size=int(final_batch), shuffle=True)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_val,dtype=torch.float32),
                                                                       torch.tensor(y_val,dtype=torch.long)),
                                         batch_size=int(final_batch), shuffle=False)

print("\nTraining final model with best hyperparameters...")
final_model.train()
for epoch in range(FINAL_EPOCHS):
    for xb, yb in train_loader:
        xb = xb.to(DEVICE); yb = yb.to(DEVICE)
        optimizer.zero_grad()
        out = final_model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
    # small val print
    final_model.eval()
    with torch.no_grad():
        corr = 0; tot = 0
        for xb, yb in val_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            logits = final_model(xb)
            preds = logits.argmax(dim=1)
            corr += (preds == yb).sum().item(); tot += yb.size(0)
    acc_val = corr / tot if tot>0 else 0.0
    print(f" Final training epoch {epoch+1}/{FINAL_EPOCHS} - Val Acc: {acc_val:.4f}")
    final_model.train()

# final prediction on user input
final_model.eval()
with torch.no_grad():
    out = final_model(user_tensor)
    probs = torch.softmax(out, dim=1).cpu().numpy().flatten()
    final_label = int(np.argmax(probs))

print("\n====================================================")
print("🔥 FINAL PREDICTION (after full training):")
print(f"Predicted probabilities -> No Fire: {probs[0]:.3f}, Fire: {probs[1]:.3f}")
if final_label == 1:
    print("🚨 FINAL RESULT: FIRE DETECTED!")
else:
    print("🌿 FINAL RESULT: NO FIRE.")
print("====================================================\n")
