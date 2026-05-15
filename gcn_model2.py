import torch, os, torch.nn.functional as F, torch, pandas as pd, re
from torch_geometric.nn import GINConv, TopKPooling, GraphNorm, global_mean_pool, global_max_pool
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

        self.conv1 = GINConv(make_mlp(in_channels, 32))
        self.gn1 = GraphNorm(32)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.pool1 = TopKPooling(32, ratio=0.5)

        self.conv2 = GINConv(make_mlp(32, 64))
        self.gn2 = GraphNorm(64)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.pool2 = TopKPooling(64, ratio=0.5)

        self.conv3 = GINConv(make_mlp(64, 128))
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
        save_path = "best_graph_gcn_model2.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model at epoch {epoch} with test accuracy = {acc:.4f}")
    print(f"Epoch {epoch:03d}, Loss={loss:.4f}, Acc={acc:.4f}, AccTrain={acc_train:.4f}")
    print(f"Recall_0={recall_0:.4f}, Recall_1={recall_1:.4f}, Recall_2={recall_2:.4f}")
    print(conf_matrix)
    print(f"Recall_0Train={recall_0_train:.4f}, Recall_1Train={recall_1_train:.4f}, Recall_2Train={recall_2_train:.4f}")
    print(conf_matrix_train)
