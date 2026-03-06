from torch_geometric.data import Data
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from rdkit import Chem
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm

import torch
import pickle

import torch.utils.data
import os

import dgl

from scipy import sparse as sp
import numpy as np

class CustomData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'line_graph_edge_index':
            return self.edge_index.size(1) if self.edge_index.nelement() != 0 else 0
        return super().__inc__(key, value, *args, **kwargs)


def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s,
                    allowable_set))

def atom_features(atom, atom_symbols, explicit_H=True, use_chirality=False):
    results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
              one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]
    if explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)


def edge_features(bond):
    bond_type = bond.GetBondType()
    return torch.tensor([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]).long()


def generate_drug_data(
    mol_graph,
    atom_symbols,
    fps_all,
    id,
    self_idx=None,
    topk=32,
    tau_mode='p70',
    d_min=8
):

    edge_list = torch.LongTensor(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()]
    )
    if edge_list.numel() > 0:
        edge_list, edge_feats = edge_list[:, :2], edge_list[:, 2:].float()

        edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0)
        edge_feats = torch.cat([edge_feats] * 2, dim=0)
    else:
        edge_list = torch.empty((0, 2), dtype=torch.long)
        edge_feats = torch.empty((0, 6), dtype=torch.float32)

    features = [(atom.GetIdx(), atom_features(atom, atom_symbols)) for atom in mol_graph.GetAtoms()]
    features.sort()
    if len(features) == 0:
        raise ValueError("Molecule has no atoms; cannot create node features.")
    _, features = zip(*features)
    features = torch.stack(features)  # [num_nodes, feat_dim]

    if edge_list.numel() > 0:
        conn = (edge_list[:, 1].unsqueeze(1) == edge_list[:, 0].unsqueeze(0)) & \
               (edge_list[:, 0].unsqueeze(1) != edge_list[:, 1].unsqueeze(0))
        line_graph_edge_index = conn.nonzero(as_tuple=False).T
        new_edge_index = edge_list.T  # [2, E]
    else:
        line_graph_edge_index = torch.empty((2, 0), dtype=torch.long)
        new_edge_index = torch.empty((2, 0), dtype=torch.long)


    N = len(fps_all)
    if self_idx is None or not (0 <= self_idx < N):

        from rdkit.Chem import AllChem
        mol_graph_fps = AllChem.GetMorganFingerprintAsBitVect(mol_graph, 2)
    else:
        mol_graph_fps = fps_all[self_idx]

    similarity_vector = torch.zeros(N, dtype=torch.float32)
    for i in range(N):
        similarity_vector[i] = DataStructs.FingerprintSimilarity(fps_all[i], mol_graph_fps)


    if self_idx is not None and 0 <= self_idx < N:
        similarity_vector[self_idx] = 0.0


    if similarity_vector.numel() == 0:
        sparse_sim = similarity_vector.clone()
    else:
        k = min(topk, similarity_vector.numel())
        topk_values, topk_indices = torch.topk(similarity_vector, k)

        positive = similarity_vector[similarity_vector > 0]
        if tau_mode == 'p70':
            tau = torch.quantile(positive, 0.70) if positive.numel() > 0 else torch.tensor(0.0)
        elif tau_mode == 'mean+0.5std':
            tau = (positive.mean() + 0.5 * positive.std()) if positive.numel() > 0 else torch.tensor(0.0)
        else:
            tau = torch.tensor(float(tau_mode))

        mask = topk_values >= tau
        kept_indices = topk_indices[mask]
        kept_values  = topk_values[mask]


        if kept_indices.numel() < d_min:
            need = d_min - kept_indices.numel()
            cand_indices = topk_indices[~mask]
            cand_values  = topk_values[~mask]
            valid = cand_values > 0
            cand_indices = cand_indices[valid]
            cand_values  = cand_values[valid]
            if cand_indices.numel() > 0:
                order = torch.argsort(cand_values, descending=True)[:need]
                kept_indices = torch.cat([kept_indices, cand_indices[order]], dim=0)
                kept_values  = torch.cat([kept_values,  cand_values[order]],  dim=0)

        sparse_sim = torch.zeros_like(similarity_vector)
        if kept_indices.numel() > 0:
            sparse_sim[kept_indices] = kept_values


    data = CustomData(
        x=features,
        edge_index=new_edge_index,
        line_graph_edge_index=line_graph_edge_index,
        edge_attr=edge_feats,
        sim=sparse_sim.unsqueeze(0),  # [1, N]
        id=id
    )
    return data



def generate_drug_data_dgl(mol_graph, atom_symbols):
    edge_list = torch.LongTensor(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()])
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (
    torch.LongTensor([]), torch.FloatTensor([]))
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats

    features = [(atom.GetIdx(), atom_features(atom, atom_symbols)) for atom in mol_graph.GetAtoms()]
    features.sort()
    _, features = zip(*features)
    features = torch.stack(features)
    node_feature = features.long()
    edge_feature = edge_feats.long()

    g = dgl.DGLGraph()
    g.add_nodes(features.shape[0])
    g.ndata['feat'] = node_feature
    for src, dst in edge_list:
        g.add_edges(src.item(), dst.item())
    g.edata['feat'] = edge_feature
    data_dgl = g
    return data_dgl


def finalize_similarity_graph(drug_data_pyg, id_to_idx, d_min=8, d_max=64, make_symmetric='min'):

    ids = list(drug_data_pyg.keys())
    N = len(ids)

    # 1) 组装稀疏矩阵 S
    rows, cols, vals = [], [], []
    for id_ in ids:
        i = id_to_idx[id_]
        sim_row = drug_data_pyg[id_].sim.squeeze(0)  # [N]
        nz = torch.nonzero(sim_row > 0, as_tuple=False).flatten()
        if nz.numel() > 0:
            rows.append(torch.full((nz.numel(),), i, dtype=torch.long))
            cols.append(nz)
            vals.append(sim_row[nz])
    if len(rows) == 0:
        return drug_data_pyg

    rows = torch.cat(rows); cols = torch.cat(cols); vals = torch.cat(vals)
    S = sp.coo_matrix((vals.numpy(), (rows.numpy(), cols.numpy())), shape=(N, N)).tocsr()


    S_T = S.transpose().tocsr()
    if make_symmetric == 'min':
        S_mut = S.minimum(S_T)
        S_mut.eliminate_zeros()  # 清理显式0
    elif make_symmetric == 'mean':

        M = S.minimum(S_T)
        M.eliminate_zeros()
        S_mut = M
    else:
        raise ValueError('make_symmetric must be "min" or "mean"')


    def row_topk_csr(A, k):
        A = A.tolil()
        for i in range(A.shape[0]):
            row_data = A.data[i]; row_idx = A.rows[i]
            if len(row_data) > k:
                order = np.argsort(row_data)[::-1][:k]
                A.data[i] = list(np.array(row_data)[order])
                A.rows[i] = list(np.array(row_idx)[order])
        return A.tocsr()

    S_mut = row_topk_csr(S_mut, d_max)


    degrees = np.diff(S_mut.indptr)
    need_fill = np.where(degrees < d_min)[0]
    if need_fill.size > 0:
        S_mut = S_mut.tolil()
        S_orig = S.tocsr()
        for i in need_fill:
            have_set = set(S_mut.rows[i])
            row = S_orig.getrow(i)  # 原始单向 TopK∧阈值结果
            if row.nnz == 0:
                continue
            order = np.argsort(row.data)[::-1]
            for idx in order:
                j = row.indices[idx]
                if j not in have_set and i != j and row.data[idx] > 0:
                    S_mut.rows[i].append(j)
                    S_mut.data[i].append(row.data[idx])
                    have_set.add(j)
                    if len(S_mut.rows[i]) >= d_min:
                        break
        S_mut = S_mut.tocsr()



    S_mut = S_mut.tocsr()
    for id_ in ids:
        i = id_to_idx[id_]
        row = S_mut.getrow(i)
        vec = torch.zeros(N, dtype=torch.float32)
        if row.nnz > 0:
            vec[row.indices] = torch.from_numpy(row.data).float()
        drug_data_pyg[id_].sim = vec.unsqueeze(0)

    return drug_data_pyg


def load_drug_mol_data(
    args,
    topk=32,
    tau_mode='p70',
    d_min=8,
    d_max=64,
    do_finalize=True
):

    df = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)
    needed_cols = [args.c_id1, args.c_id2, args.c_s1, args.c_s2, args.c_y]
    df = df[needed_cols].copy()


    drug_smile_dict = {}
    for id1, id2, smi1, smi2, _ in zip(df[args.c_id1], df[args.c_id2], df[args.c_s1], df[args.c_s2], df[args.c_y]):
        if id1 not in drug_smile_dict:
            drug_smile_dict[id1] = smi1
        if id2 not in drug_smile_dict:
            drug_smile_dict[id2] = smi2


    drug_id_mol_tup = []   # [(id, mol)]
    symbols = []
    for did, smi in drug_smile_dict.items():
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol is None:
            continue
        drug_id_mol_tup.append((did, mol))
        symbols.extend(atom.GetSymbol() for atom in mol.GetAtoms())
    symbols = list(set(symbols))


    drug_id_mol_tup.sort(key=lambda x: str(x[0]))


    id_to_idx = {did: idx for idx, (did, _) in enumerate(drug_id_mol_tup)}
    fps_all = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for _, mol in drug_id_mol_tup]


    drug_data_pyg = {}
    for did, mol in tqdm(drug_id_mol_tup, desc='Processing drugs_pyg'):
        self_idx = id_to_idx[did]
        data_i = generate_drug_data(
            mol_graph=mol,
            atom_symbols=symbols,
            fps_all=fps_all,
            id=did,
            self_idx=self_idx,
            topk=topk,
            tau_mode=tau_mode,
            d_min=d_min
        )
        drug_data_pyg[did] = data_i


    drug_data_dgl = {did: generate_drug_data_dgl(mol, symbols)
                     for did, mol in tqdm(drug_id_mol_tup, desc='Processing drugs_dgl')}


    if do_finalize and 'finalize_similarity_graph' in globals():
        drug_data_pyg = finalize_similarity_graph(
            drug_data_pyg, id_to_idx, d_min=d_min, d_max=d_max, make_symmetric='min'
        )


    save_data(drug_data_pyg, 'drug_data_pyg.pkl', args)
    save_data(drug_data_dgl, 'drug_data_dgl.pkl', args)

    return drug_data_pyg, drug_data_dgl



def generate_pair_triplets(args):
    pos_triplets = []

    with open(f'{args.dirname}/{args.dataset.lower()}/drug_data_pyg.pkl', 'rb') as f:
        drug_ids = list(pickle.load(f).keys())

    data = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)
    for id1, id2, relation in zip(data[args.c_id1], data[args.c_id2], data[args.c_y]):
        if ((id1 not in drug_ids) or (id2 not in drug_ids)): continue
        if args.dataset in ('drugbank',):
            relation -= 1
        pos_triplets.append([id1, id2, relation])

    if len(pos_triplets) == 0:
        raise ValueError('All tuples are invalid.')

    pos_triplets = np.array(pos_triplets)
    data_statistics = load_data_statistics(pos_triplets)
    drug_ids = np.array(drug_ids)

    neg_samples = []
    for pos_item in tqdm(pos_triplets, desc='Generating Negative sample'):
        temp_neg = []
        h, t, r = pos_item[:3]

        if args.dataset == 'drugbank':
            neg_heads, neg_tails = _normal_batch(h, t, r, args.neg_ent, data_statistics, drug_ids, args)
            temp_neg = [str(neg_h) + '$h' for neg_h in neg_heads] + \
                       [str(neg_t) + '$t' for neg_t in neg_tails]
        else:
            existing_drug_ids = np.asarray(list(set(
                np.concatenate(
                    [data_statistics["ALL_TRUE_T_WITH_HR"][(h, r)], data_statistics["ALL_TRUE_H_WITH_TR"][(h, r)]],
                    axis=0)
            )))
            temp_neg = _corrupt_ent(existing_drug_ids, args.neg_ent, drug_ids, args)

        neg_samples.append('_'.join(map(str, temp_neg[:args.neg_ent])))

    df = pd.DataFrame({'Drug1_ID': pos_triplets[:, 0],
                       'Drug2_ID': pos_triplets[:, 1],
                       'Y': pos_triplets[:, 2],
                       'Neg samples': neg_samples})
    filename = f'{args.dirname}/{args.dataset}/pair_pos_neg_triplets.csv'
    df.to_csv(filename, index=False)
    print(f'\nData saved as {filename}!')
    save_data(data_statistics, 'data_statistics.pkl', args)


def load_data_statistics(all_tuples):
    print('Loading data statistics ...')
    statistics = dict()
    statistics["ALL_TRUE_H_WITH_TR"] = defaultdict(list)
    statistics["ALL_TRUE_T_WITH_HR"] = defaultdict(list)
    statistics["FREQ_REL"] = defaultdict(int)
    statistics["ALL_H_WITH_R"] = defaultdict(dict)
    statistics["ALL_T_WITH_R"] = defaultdict(dict)
    statistics["ALL_TAIL_PER_HEAD"] = {}
    statistics["ALL_HEAD_PER_TAIL"] = {}

    for h, t, r in tqdm(all_tuples, desc='Getting data statistics'):
        statistics["ALL_TRUE_H_WITH_TR"][(t, r)].append(h)
        statistics["ALL_TRUE_T_WITH_HR"][(h, r)].append(t)
        statistics["FREQ_REL"][r] += 1.0
        statistics["ALL_H_WITH_R"][r][h] = 1
        statistics["ALL_T_WITH_R"][r][t] = 1

    for t, r in statistics["ALL_TRUE_H_WITH_TR"]:
        statistics["ALL_TRUE_H_WITH_TR"][(t, r)] = np.array(list(set(statistics["ALL_TRUE_H_WITH_TR"][(t, r)])))
    for h, r in statistics["ALL_TRUE_T_WITH_HR"]:
        statistics["ALL_TRUE_T_WITH_HR"][(h, r)] = np.array(list(set(statistics["ALL_TRUE_T_WITH_HR"][(h, r)])))

    for r in statistics["FREQ_REL"]:
        statistics["ALL_H_WITH_R"][r] = np.array(list(statistics["ALL_H_WITH_R"][r].keys()))
        statistics["ALL_T_WITH_R"][r] = np.array(list(statistics["ALL_T_WITH_R"][r].keys()))
        statistics["ALL_HEAD_PER_TAIL"][r] = statistics["FREQ_REL"][r] / len(statistics["ALL_T_WITH_R"][r])
        statistics["ALL_TAIL_PER_HEAD"][r] = statistics["FREQ_REL"][r] / len(statistics["ALL_H_WITH_R"][r])

    print('getting data statistics done!')

    return statistics


def _corrupt_ent(positive_existing_ents, max_num, drug_ids, args):
    corrupted_ents = []
    while len(corrupted_ents) < max_num:
        candidates = args.random_num_gen.choice(drug_ids, (max_num - len(corrupted_ents)) * 2, replace=False)
        invalid_drug_ids = np.concatenate([positive_existing_ents, corrupted_ents], axis=0)
        mask = np.isin(candidates, invalid_drug_ids, assume_unique=True, invert=True)
        corrupted_ents.extend(candidates[mask])

    corrupted_ents = np.array(corrupted_ents)[:max_num]
    return corrupted_ents


def _normal_batch(h, t, r, neg_size, data_statistics, drug_ids, args):
    neg_size_h = 0
    neg_size_t = 0
    prob = data_statistics["ALL_TAIL_PER_HEAD"][r] / (data_statistics["ALL_TAIL_PER_HEAD"][r] +
                                                      data_statistics["ALL_HEAD_PER_TAIL"][r])

    for i in range(neg_size):
        if args.random_num_gen.random() < prob:
            neg_size_h += 1
        else:
            neg_size_t += 1

    return (_corrupt_ent(data_statistics["ALL_TRUE_H_WITH_TR"][t, r], neg_size_h, drug_ids, args),
            _corrupt_ent(data_statistics["ALL_TRUE_T_WITH_HR"][h, r], neg_size_t, drug_ids, args))


def save_data(data, filename, args):
    dirname = f'{args.dirname}/{args.dataset}'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = dirname + '/' + filename
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nData saved as {filename}!')


def split_data(args):
    filename = f'{args.dirname}/{args.dataset}/pair_pos_neg_triplets.csv'
    df = pd.read_csv(filename)
    seed = args.seed
    class_name = args.class_name
    save_to_filename = os.path.splitext(filename)[0]
    cv_split = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=seed)
    for fold_i, (train_index, test_index) in enumerate(cv_split.split(X=df, y=df[class_name])):
        print(f'Fold {fold_i} generated!')
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        train_df.to_csv(f'{save_to_filename}_train_fold{fold_i}.csv', index=False)
        print(f'{save_to_filename}_train_fold{fold_i}.csv', 'saved!')
        test_df.to_csv(f'{save_to_filename}_test_fold{fold_i}.csv', index=False)
        print(f'{save_to_filename}_test_fold{fold_i}.csv', 'saved!')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, required=True, choices=['drugbank'],
                        help='Dataset to preprocess.')
    parser.add_argument('-n', '--neg_ent', type=int, default=1, help='Number of negative samples')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Seed for the random number generator')
    parser.add_argument('-o', '--operation', type=str, required=True,
                        choices=['all', 'generate_triplets', 'drug_data', 'split'], help='Operation to perform')
    parser.add_argument('-t_r', '--test_ratio', type=float, default=0.2)
    parser.add_argument('-n_f', '--n_folds', type=int, default=3)

    dataset_columns_map = {
        'drugbank': ('ID1', 'ID2', 'X1', 'X2', 'Y'),
        'twosides': ('Drug1_ID', 'Drug2_ID', 'Drug1', 'Drug2', 'New Y'),
    }

    dataset_file_name_map = {
#1
        'drugbank': ('./data/drugbank.tab', '\t')
    }
    args = parser.parse_args()
    args.dataset = args.dataset.lower()

    args.c_id1, args.c_id2, args.c_s1, args.c_s2, args.c_y = dataset_columns_map[args.dataset]
    args.dataset_filename, args.delimiter = dataset_file_name_map[args.dataset]
    args.dirname = '/tmp/SRR-DDI/DrugBank/data/warm start'


    args.random_num_gen = np.random.RandomState(args.seed)
    if args.operation in ('all', 'drug_data'):
        load_drug_mol_data(args)

    if args.operation in ('all', 'generate_triplets'):
        generate_pair_triplets(args)

    if args.operation in ('all', 'split'):
        args.class_name = 'Y'
        split_data(args)
