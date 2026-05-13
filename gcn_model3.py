import torch, os, torch.nn.functional as F, torch, pandas as pd, re
from torch_geometric.nn import GCNConv, GINConv, TopKPooling, GraphNorm, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from sklearn.model_selection import train_test_split
import joblib
print("CUDA available:", torch.cuda.is_available(), flush=True)
print("Num devices:", torch.cuda.device_count(), flush=True)
labels = pd.read_csv("LABELS.csv").set_index("Subject")
labels = labels[~labels.index.duplicated(keep='first')]
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0), flush=True)

filenames = os.listdir("/csl/users/2026lzhu/PRELIM_GRAPHS")

x, y = pd.Series(), pd.Series()
for fname in filenames:
    x = pd.concat([x, pd.Series([fname])])
    subjnum = int(re.search("^\d+", fname).group())
    group = {"PD":2, "Prodromal":1, "Control":0}[labels.loc[subjnum]['Group']]
    y = pd.concat([y, pd.Series([group])])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
'''
class GraphFileDataset(Dataset):
    def __init__(self, graph_names, graph_dir, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform) 
        self.graph_dir = graph_dir
        self.files = sorted(graph_names)
    
    def len(self):
        return len(self.files)
    
    def get(self, idx):
        fname = self.files[idx]
        graph = torch.load(os.path.join(self.graph_dir, fname), weights_only=False)
        graph.edge_index = graph.edge_index.long()
        subjnum = int(re.search("^\d+", fname).group())
        group = {"PD":2, "Prodromal":1, "Control":0}[labels.loc[subjnum]['Group']]
        graph.y = torch.tensor([group], dtype=torch.long)
        return graph
'''
def make_mlp(in_dim, out_dim):
    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, out_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(out_dim, out_dim)
    )
    return model

class GraphGCN(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = GCNConv(in_channels, 32)
        self.gn1 = GraphNorm(32)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.pool1 = TopKPooling(32, ratio=0.5)

        self.conv2 = GCNConv(32, 64)
        self.gn2 = GraphNorm(64)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.pool2 = TopKPooling(64, ratio=0.5)

        self.conv3 = GCNConv(64, 128)
        self.gn3 = GraphNorm(128)

        self.lin1 = torch.nn.Linear(448, 32)
        self.lin2 = torch.nn.Linear(32, 16)
        self.lin_out = torch.nn.Linear(16, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.gn1(x)
        x = F.leaky_relu(x)
        x = self.dropout1(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)
        x1 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        x = self.conv2(x, edge_index)
        x = self.gn2(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)
        x2 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        x = self.conv3(x, edge_index)
        x = self.gn3(x)
        x = F.leaky_relu(x)
        x3 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        x = torch.cat([x1, x2, x3], dim=1)

        x = F.leaky_relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin_out(x)

        return F.log_softmax(x, dim=-1)
'''
def train(model, loader, optimizer, device, accumulation_steps):
    model.train()
    total_loss = 0
    batch_idx = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y, weight=torch.tensor([1.0, 1.0, 1.0], device=device))
        loss /= accumulation_steps
        loss.backward()
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        batch_idx += 1
        total_loss += loss.item() * data.num_graphs * accumulation_steps
    if (batch_idx) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    correct = 0
    recalls = [0, 0, 0]
    counts = [0, 0, 0]
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        for i in range(len(data.y)):
            counts[int(data.y[i])] += 1
            if int(pred[i]) == int(data.y[i]): recalls[int(data.y[i])] += 1
        for t, p in zip(data.y, pred):
            conf_matrix[int(t), int(p)] += 1
    return correct / len(loader.dataset), recalls[0] / counts[0], recalls[1] / counts[1], recalls[2] / counts[2], conf_matrix

num_classes = 3

train_dataset = GraphFileDataset(x_train, "/csl/users/2026lzhu/PRELIM_GRAPHS")
train_loader = DataLoader(
    train_dataset,
    batch_size=10,
    shuffle=True,
    pin_memory=False,
    num_workers=0
)

test_dataset = GraphFileDataset(x_test, "/csl/users/2026lzhu/PRELIM_GRAPHS")
test_loader = DataLoader(
    test_dataset,
    batch_size=10,
    shuffle=True,
    pin_memory=False,
    num_workers=0
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GraphGCN(in_channels=7,
                  num_classes=num_classes).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)

max_acc = 0
for epoch in range(1, 101):
    loss = train(model, train_loader, optimizer, device, 2)
    acc, recall_0, recall_1, recall_2, conf_matrix = test(model, test_loader, device)
    acc_train, recall_0_train, recall_1_train, recall_2_train, conf_matrix_train = test(model, train_loader, device)
    if acc > max_acc: 
        max_acc = acc   
        save_path = "best_graph_gcn_modelM.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model at epoch {epoch} with test accuracy = {acc:.4f}")
    print(f"Epoch {epoch:03d}, Loss={loss:.4f}, Acc={acc:.4f}, AccTrain={acc_train:.4f}")
    print(f"Recall_0={recall_0:.4f}, Recall_1={recall_1:.4f}, Recall_2={recall_2:.4f}")
    print(conf_matrix)
    print(f"Recall_0Train={recall_0_train:.4f}, Recall_1Train={recall_1_train:.4f}, Recall_2Train={recall_2_train:.4f}")
    print(conf_matrix_train)
'''

class SimpleNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(6, 9)
        self.lin2 = torch.nn.Linear(9, 5)
        self.lin3 = torch.nn.Linear(5, 3)

    def forward(self, x):
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 3
model = SimpleNN()
gcn_model = GraphGCN(in_channels=7,
                  num_classes=num_classes).to(device)
gcn_model.load_state_dict(torch.load("best_graph_gcn_modelM.pt"))
lr_model = joblib.load("/csl/users/2026lzhu/lr_model.joblib")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr_info = pd.read_csv("/csl/users/2026lzhu/logregimptest.csv"); lr_info = pd.concat([lr_info, pd.read_csv("/csl/users/2026lzhu/logregimptrain.csv")])
ohe_features = ["EVENT_ID=BL","EVENT_ID=R01","EVENT_ID=R04","EVENT_ID=R06","EVENT_ID=R08","EVENT_ID=R10","EVENT_ID=R12","EVENT_ID=R13","EVENT_ID=R14","EVENT_ID=R15","EVENT_ID=R16","EVENT_ID=R17","EVENT_ID=R18","EVENT_ID=R19","EVENT_ID=R20","EVENT_ID=R21","EVENT_ID=SC","EVENT_ID=ST","EVENT_ID=V01","EVENT_ID=V02","EVENT_ID=V03","EVENT_ID=V04","EVENT_ID=V05","EVENT_ID=V06","EVENT_ID=V07","EVENT_ID=V08","EVENT_ID=V09","EVENT_ID=V10","EVENT_ID=V11","EVENT_ID=V12","EVENT_ID=V13","EVENT_ID=V14","EVENT_ID=V15","EVENT_ID=V16","EVENT_ID=V17","EVENT_ID=V18","EVENT_ID=V19","EVENT_ID=V20","EVENT_ID=V21","EVENT_ID=V22"]
lr_info = lr_info.set_index("PATNO")
x_train = pd.Series([fname for fname in x_train if int(re.search("^\d+", fname).group()) in lr_info.index])
x_test = pd.Series([fname for fname in x_test if int(re.search("^\d+", fname).group()) in lr_info.index])

class TwoModelDataset(Dataset):
    def __init__(self, graph_names, graph_dir, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform) 
        self.graph_dir = graph_dir
        self.files = sorted(graph_names)

    def len(self):
        return len(self.files)
    
    def get(self, idx):
        fname = self.files[idx]
        graph = torch.load(os.path.join(self.graph_dir, fname), weights_only=False)
        graph.edge_index = graph.edge_index.long()
        gcn_output = None
        with torch.no_grad():
            gcn_output = torch.exp(gcn_model(graph.x, graph.edge_index, batch=None))
        subjnum = int(re.search("^\d+", fname).group())
        lr_info_pat = lr_info.loc[[subjnum]]; lr_info_pat = lr_info_pat.drop(columns=['OUTCOME'])
        lr_info_ohe = lr_info_pat[ohe_features]; lr_info_nohe = lr_info_pat.drop(columns=ohe_features)
        lr_info_ohe = lr_info_ohe.mode().iloc[0]
        lr_info_nohe = lr_info_nohe.mean()
        lr_input = pd.concat([lr_info_nohe, lr_info_ohe]).to_frame().T
        group = {"PD":2, "Prodromal":1, "Control":0}[labels.loc[subjnum]['Group']]
        lr_output = torch.tensor(lr_model.predict_proba(lr_input), dtype=torch.float32)
        combined_features = torch.cat([gcn_output, lr_output], dim=1)
        return Data(x=combined_features, y=group)

def train_spec(model, loader, optimizer, device, accumulation_steps):
    model.train()
    total_loss = 0
    batch_idx = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x)
        loss = F.nll_loss(out, data.y, weight=torch.tensor([1.0, 1.0, 1.0], device=device))
        loss /= accumulation_steps
        loss.backward()
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        batch_idx += 1
        total_loss += loss.item() * data.num_graphs * accumulation_steps
    if (batch_idx) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(loader.dataset)

@torch.no_grad()
def test(model, loader, device):
    model.eval()
    correct = 0
    recalls = [0, 0, 0]
    counts = [0, 0, 0]
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for data in loader:
        data = data.to(device)
        out = model(data.x)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        for i in range(len(data.y)):
            counts[int(data.y[i])] += 1
            if int(pred[i]) == int(data.y[i]): recalls[int(data.y[i])] += 1
        for t, p in zip(data.y, pred):
            conf_matrix[int(t), int(p)] += 1
    return correct / len(loader.dataset), recalls[0] / counts[0], recalls[1] / counts[1], recalls[2] / counts[2], conf_matrix


train_dataset = TwoModelDataset(x_train, "/csl/users/2026lzhu/PRELIM_GRAPHS")
train_loader = DataLoader(
    train_dataset,
    batch_size=10,
    shuffle=True,
    pin_memory=False,
    num_workers=0
)

test_dataset = TwoModelDataset(x_test, "/csl/users/2026lzhu/PRELIM_GRAPHS")
test_loader = DataLoader(
    test_dataset,
    batch_size=10,
    shuffle=True,
    pin_memory=False,
    num_workers=0
)
max_acc = 0
for epoch in range(1, 101):
    loss = train_spec(model, train_loader, optimizer, device, 2)
    acc, recall_0, recall_1, recall_2, conf_matrix = test(model, test_loader, device)
    acc_train, recall_0_train, recall_1_train, recall_2_train, conf_matrix_train = test(model, train_loader, device)
    if acc > max_acc: 
        max_acc = acc   
        save_path = "best_graph_gcn_modelM*^.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model at epoch {epoch} with test accuracy = {acc:.4f}")
    print(f"Epoch {epoch:03d}, Loss={loss:.4f}, Acc={acc:.4f}, AccTrain={acc_train:.4f}")
    print(f"Recall_0={recall_0:.4f}, Recall_1={recall_1:.4f}, Recall_2={recall_2:.4f}")
    print(conf_matrix)
    print(f"Recall_0Train={recall_0_train:.4f}, Recall_1Train={recall_1_train:.4f}, Recall_2Train={recall_2_train:.4f}")
    print(conf_matrix_train)