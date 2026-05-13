import nibabel as nib, matplotlib.pyplot as plt, pandas as pd, numpy as np, os, torch, re, torch_geometric, random
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
labels = pd.read_csv("/csl/users/2026lzhu/LABELS.csv").set_index("Subject")
labels = labels[~labels.index.duplicated(keep='first')]
dir_list= os.listdir("/csl/users/2026lzhu/CORT_SURFACES3")
direc = "/csl/users/2026lzhu/CORT_SURFACES3"
from scipy.ndimage import gaussian_filter
import torch
import numpy as np
import random
random.seed(42)
np.random.seed(42)

def augment_edges(edge_index, num_nodes, rewire_ratio=0.05):
    edges = edge_index.cpu().numpy().astype(np.int64)
    E = edges.shape[1]
    existing = {}
    for i in range(edges.shape[1]):
        u, v = int(edges[0, i]), int(edges[1, i])
        if u != v:
            a, b = (u, v)
            if a in existing: existing[a].add(b)
            else: existing[a] = {b}
    print(len(existing))
    print(num_nodes)
    num_rewire = int(E * rewire_ratio)
    new_edges = []
    attempts = 0
    max_attempts = num_rewire * 20
    while len(new_edges) < num_rewire and attempts < max_attempts:
        u = random.randint(0, num_nodes-1)
        while len(existing[u]) <= 1: u = random.randint(0, num_nodes-1)
        to_rewire = random.randint(1,len(existing[u]))    
        for i in range(to_rewire):
            node = random.randint(0,num_nodes-1)
            while node == u: node = random.randint(0,num_nodes-1)
            old = existing[u].pop()
            existing[u].add(node)
            existing[old].remove(u)
            if node in existing: existing[node].add(u)
            else: existing[node] = {u}
        attempts += 1
    for u in existing:
        for n in existing[u]:
            new_edges.append((u, n))
    if len(new_edges) > 0:
        new_edges = np.array(new_edges, dtype=np.int64).T
        edges_new = np.hstack([new_edges])
    else:
        edges_new = edges
    edges_new = np.unique(edges_new.astype(np.int64), axis=1)
    if edges_new.shape[0] != 2:
        edges_new = edges_new.T
    return torch.tensor(edges_new, dtype=torch.long)

for dr in dir_list:
    subjnum = int(re.search("^\d+", dr).group())
    if labels.loc[subjnum]["Group"] == "SWEDD": continue
    label = {"Control":0,"Prodromal":1,"PD":2}[labels.loc[subjnum]["Group"]]
    lh_pial_path = f"{direc}/{dr}/surf/lh.pial.T1"
    lh_curv_path = f"{direc}/{dr}/surf/lh.curv"
    lh_sulc_path = f"{direc}/{dr}/surf/lh.sulc"
    lh_thick_path = f"{direc}/{dr}/surf/lh.thickness"
    lh_area_path = f"{direc}/{dr}/surf/lh.area"
    rh_pial_path = f"{direc}/{dr}/surf/rh.pial.T1"
    rh_curv_path = f"{direc}/{dr}/surf/rh.curv"
    rh_sulc_path = f"{direc}/{dr}/surf/rh.sulc"
    rh_thick_path = f"{direc}/{dr}/surf/rh.thickness"
    rh_area_path = f"{direc}/{dr}/surf/rh.area"

    rvertices, rfaces = nib.freesurfer.read_geometry(rh_pial_path)
    rfaces = rfaces.astype("<i4")
    rcurv = nib.freesurfer.read_morph_data(rh_curv_path)
    rsulc = nib.freesurfer.read_morph_data(rh_sulc_path)
    rthick = nib.freesurfer.read_morph_data(rh_thick_path)
    rarea = nib.freesurfer.read_morph_data(rh_area_path)
    lvertices, lfaces = nib.freesurfer.read_geometry(lh_pial_path)
    lfaces = lfaces.astype("<i4")
    lcurv = nib.freesurfer.read_morph_data(lh_curv_path)
    lsulc = nib.freesurfer.read_morph_data(lh_sulc_path)
    lthick = nib.freesurfer.read_morph_data(lh_thick_path)
    larea = nib.freesurfer.read_morph_data(lh_area_path)

    lfaces_torch = torch.tensor(lfaces, dtype=torch.long)
    ledges = torch.cat([
        lfaces_torch[:, [0, 1]],
        lfaces_torch[:, [1, 2]],
        lfaces_torch[:, [2, 0]]
    ], dim=0).long().t().contiguous()

    ledges = to_undirected(ledges)
    ledges = torch.unique(ledges, dim=1).contiguous()    

    lx = torch.tensor(np.vstack([lvertices.T, lcurv, lsulc, lthick, larea]).T, dtype=torch.float) 
    lstatic = np.vstack([lcurv, lsulc, lthick, larea])
    ldata = Data(x=lx, edge_index=ledges)

    rfaces_torch = torch.tensor(rfaces, dtype=torch.long)
    redges = torch.cat([
        rfaces_torch[:, [0, 1]],
        rfaces_torch[:, [1, 2]],
        rfaces_torch[:, [2, 0]]
    ], dim=0).long().t().contiguous()

    redges = to_undirected(redges)
    redges = torch.unique(redges, dim=1).contiguous()

    rx = torch.tensor(np.vstack([rvertices.T, rcurv, rsulc, rthick, rarea]).T, dtype=torch.float) 
    rstatic = np.vstack([rcurv, rsulc, rthick, rarea])
    rdata = Data(x=rx, edge_index=redges)

    offset = ldata.num_nodes
    full_graph_x = torch.cat([ldata.x, rdata.x], dim=0)
    full_graph_edge_index = torch.cat([ldata.edge_index, rdata.edge_index+offset], dim=1)
    data = Data(x=full_graph_x, edge_index=full_graph_edge_index)
    print(data.num_nodes, data.num_edges)
    torch.save(data, f"/csl/users/2026lzhu/PRELIM_GRAPHS/{dr}_{label}.pt")
    if label == 1:
        if random.randint(1,8) == 8:
            lnoise = np.random.normal(0, 0.05, size=lvertices.shape)
            lverts_noise = lvertices+lnoise
            rnoise = np.random.normal(0, 0.05, size=rvertices.shape)
            rverts_noise = rvertices+rnoise

            theta = np.radians(np.random.uniform(-5,5))
            phi = np.radians(np.random.uniform(-5,5))
            scale = np.random.uniform(0.95, 1.05)
            Rx = np.array([[1,0,0], [0,np.cos(theta),-np.sin(theta)], [0,np.sin(theta),np.cos(theta)]])
            Rz = np.array([[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1]])
            totR = Rx @ Rz
            lrotscale = (lverts_noise @ totR.T)*scale
            rrotscale = (rverts_noise @ totR.T)*scale

            rdisp = np.random.randn(*rvertices.shape)
            rdisp = gaussian_filter(rdisp, sigma=3, mode="reflect")
            rfinal = rrotscale + 0.1*rdisp
            ldisp = np.random.randn(*lvertices.shape)
            ldisp = gaussian_filter(ldisp, sigma=3, mode="reflect")
            lfinal = lrotscale + 0.1*ldisp

            lx = torch.tensor(np.vstack([lfinal.T, lstatic]).T, dtype=torch.float)
            rx = torch.tensor(np.vstack([rfinal.T, rstatic]).T, dtype=torch.float)
            ledges_aug = augment_edges(ledges, lx.shape[0], 0.05)
            redges_aug = augment_edges(redges, rx.shape[0], 0.05)
            
            ldata = Data(x=lx, edge_index=ledges_aug)
            rdata = Data(x=rx, edge_index=redges_aug)
            offset = ldata.num_nodes
            full_graph_x = torch.cat([ldata.x, rdata.x], dim=0)
            full_graph_edge_index = torch.cat([ldata.edge_index, rdata.edge_index+offset], dim=1)
            data = Data(x=full_graph_x, edge_index=full_graph_edge_index)
            print(data.num_nodes, data.num_edges)
            #torch.save(data, f"/csl/users/2026lzhu/PRELIM_GRAPHS/{dr}_{label}_1.pt")
    elif label == 0: 
        for i in range(2):
            lnoise = np.random.normal(0, 0.05, size=lvertices.shape)
            lverts_noise = lvertices+lnoise
            rnoise = np.random.normal(0, 0.05, size=rvertices.shape)
            rverts_noise = rvertices+rnoise

            theta = np.radians(np.random.uniform(-5,5))
            phi = np.radians(np.random.uniform(-5,5))
            scale = np.random.uniform(0.95, 1.05)
            Rx = np.array([[1,0,0], [0,np.cos(theta),-np.sin(theta)], [0,np.sin(theta),np.cos(theta)]])
            Rz = np.array([[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1]])
            totR = Rx @ Rz
            lrotscale = (lverts_noise @ totR.T)*scale
            rrotscale = (rverts_noise @ totR.T)*scale

            rdisp = np.random.randn(*rvertices.shape)
            rdisp = gaussian_filter(rdisp, sigma=3, mode="reflect")
            rfinal = rrotscale + 0.1*rdisp
            ldisp = np.random.randn(*lvertices.shape)
            ldisp = gaussian_filter(ldisp, sigma=3, mode="reflect")
            lfinal = lrotscale + 0.1*ldisp

            lx = torch.tensor(np.vstack([lfinal.T, lstatic]).T, dtype=torch.float)
            rx = torch.tensor(np.vstack([rfinal.T, rstatic]).T, dtype=torch.float)
            ledges_aug = augment_edges(ledges, lx.shape[0], 0.05)
            redges_aug = augment_edges(redges, rx.shape[0], 0.05)
            
            ldata = Data(x=lx, edge_index=ledges_aug)
            rdata = Data(x=rx, edge_index=redges_aug)
            offset = ldata.num_nodes
            full_graph_x = torch.cat([ldata.x, rdata.x], dim=0)
            full_graph_edge_index = torch.cat([ldata.edge_index, rdata.edge_index+offset], dim=1)
            data = Data(x=full_graph_x, edge_index=full_graph_edge_index)
            print(data.num_nodes, data.num_edges)
            #torch.save(data, f"/csl/users/2026lzhu/PRELIM_GRAPHS/{dr}_{label}_{i+1}.pt")
    elif label == 2:
        for i in range(1):
            lnoise = np.random.normal(0, 0.05, size=lvertices.shape)
            lverts_noise = lvertices+lnoise
            rnoise = np.random.normal(0, 0.05, size=rvertices.shape)
            rverts_noise = rvertices+rnoise

            theta = np.radians(np.random.uniform(-5,5))
            phi = np.radians(np.random.uniform(-5,5))
            scale = np.random.uniform(0.95, 1.05)
            Rx = np.array([[1,0,0], [0,np.cos(theta),-np.sin(theta)], [0,np.sin(theta),np.cos(theta)]])
            Rz = np.array([[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1]])
            totR = Rx @ Rz
            lrotscale = (lverts_noise @ totR.T)*scale
            rrotscale = (rverts_noise @ totR.T)*scale

            rdisp = np.random.randn(*rvertices.shape)
            rdisp = gaussian_filter(rdisp, sigma=3, mode="reflect")
            rfinal = rrotscale + 0.1*rdisp
            ldisp = np.random.randn(*lvertices.shape)
            ldisp = gaussian_filter(ldisp, sigma=3, mode="reflect")
            lfinal = lrotscale + 0.1*ldisp

            lx = torch.tensor(np.vstack([lfinal.T, lstatic]).T, dtype=torch.float)
            rx = torch.tensor(np.vstack([rfinal.T, rstatic]).T, dtype=torch.float)
            ledges_aug = augment_edges(ledges, lx.shape[0], 0.05)
            redges_aug = augment_edges(redges, rx.shape[0], 0.05)
            
            ldata = Data(x=lx, edge_index=ledges_aug)
            rdata = Data(x=rx, edge_index=redges_aug)
            offset = ldata.num_nodes
            full_graph_x = torch.cat([ldata.x, rdata.x], dim=0)
            full_graph_edge_index = torch.cat([ldata.edge_index, rdata.edge_index+offset], dim=1)
            data = Data(x=full_graph_x, edge_index=full_graph_edge_index)
            print(data.num_nodes, data.num_edges)
            torch.save(data, f"/csl/users/2026lzhu/PRELIM_GRAPHS/{dr}_{label}_{i+1}.pt")