import argparse
import os
import glob
import torch
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from dataset import load_ddi_dataset
from model import gnn_model
from data_pre import CustomData
from torch_geometric.data import Batch


def load_model(ckpt_path, net_params, device):
    model = gnn_model('GraphTransformer', net_params)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


from rdkit.Chem import rdDepictor

def visualize_attention(smiles, weights, save_path):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[WARN] Failed to parse SMILES: {smiles}")
        return
    rdDepictor.Compute2DCoords(mol)

    conf = mol.GetConformer()
    fig = Draw.MolToMPL(mol, size=(300, 300), kekulize=True)
    ax = fig.axes[0]

    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        ax.text(pos.x, pos.y, f'{weights[i]:.2f}', fontsize=8, ha='center', va='center', color='red')

    fig.savefig(save_path)
    plt.close(fig)



def build_line_graph_edges(edge_index):
    src, dst = edge_index
    num_edges = edge_index.size(1)
    connections = []
    for i in range(num_edges):
        for j in range(num_edges):
            if dst[i].item() == src[j].item():
                connections.append((i, j))
    if not connections:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(connections, dtype=torch.long).t()


def add_attention_fields(batch_graph, device):
    new_data_list = []
    for g in batch_graph.to_data_list():
        g = g.clone()
        g.edge_index_batch = torch.zeros(g.edge_index.size(1), dtype=torch.long, device=device)
        g.line_graph_edge_index = build_line_graph_edges(g.edge_index).to(device)
        new_data_list.append(g)
    return Batch.from_data_list(new_data_list).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./attention_output')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--data_root', type=str, required=True, help='Path to preprocessed graph data')
    parser.add_argument('--source', type=str, choices=['self', 'gated'], default='self', help='Attention source to visualize')

    args = parser.parse_args()

    if args.ckpt_path is None and args.ckpt_dir:
        pt_files = glob.glob(os.path.join(args.ckpt_dir, 'epoch-*.pt'))
        if not pt_files:
            raise FileNotFoundError(f"No checkpoints found in {args.ckpt_dir}")
        args.ckpt_path = max(pt_files, key=os.path.getmtime)
        print(f"[INFO] Using latest checkpoint: {args.ckpt_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = load_ddi_dataset(args.data_root, batch_size=args.batch_size, fold=args.fold)
    loader = {'train': train_loader, 'val': val_loader, 'test': test_loader}[args.set]

    data = next(iter(loader))
    head, tail, head_dgl, tail_dgl, rel, label = data

    head = Batch.from_data_list(head.to_data_list()).to(device)
    tail = Batch.from_data_list(tail.to_data_list()).to(device)

    rel = rel.to(device)
    label = label.to(device)

    head_dgl = head_dgl.to(device)
    tail_dgl = tail_dgl.to(device)
    head_dgl.edata['feat'] = head_dgl.edata['feat'].to(device)
    tail_dgl.edata['feat'] = tail_dgl.edata['feat'].to(device)

    head = add_attention_fields(head, device)
    tail = add_attention_fields(tail, device)

    node_dim = head.x.size(-1)
    edge_dim = head.edge_attr.size(-1)

    net_params = dict(
        L=2,
        n_heads=6,
        hidden_dim=96,
        out_dim=96,
        edge_feat=True,
        residual=True,
        readout="mean",
        in_feat_dropout=0.2,
        dropout=0.2,
        layer_norm=False,
        batch_norm=True,
        self_loop=False,
        lap_pos_enc=True,
        pos_enc_dim=6,
        full_graph=False,
        batch_size=args.batch_size,
        num_atom_type=node_dim,
        num_bond_type=edge_dim,
        device=device,
        n_iter=10,
        use_self_attention=(args.source == 'self')
    )

    model = load_model(args.ckpt_path, net_params, device)

    os.makedirs(args.save_dir, exist_ok=True)

    with torch.no_grad():
        model.forward(head, tail, head_dgl, tail_dgl,
                      head_dgl.edata['feat'], tail_dgl.edata['feat'],
                      rel, head.sim.to(device), tail.sim.to(device))

        att_weights = model.drug_encoder.line_graph.att.att_weights
        if att_weights is None:
            raise RuntimeError("No attention weights found. Ensure the model's attention layer is correctly storing them.")

        smiles_list = [g.smiles for g in head.to_data_list()]
        att_weights = att_weights.squeeze().cpu().numpy()

        start_idx = 0
        for i, mol in enumerate(smiles_list[:args.num]):
            num_atoms = head.ptr[i + 1] - head.ptr[i]
            w = att_weights[start_idx:start_idx + num_atoms]
            save_path = os.path.join(args.save_dir, f'{args.set}_sample{i}.png')
            visualize_attention(mol, w, save_path)
            start_idx += num_atoms

        print(f"[INFO] Saved {args.num} attention visualizations to: {args.save_dir}")


if __name__ == '__main__':
    main()
