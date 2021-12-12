import pandas as pd
from collections import Counter

import torch 
from torch_geometric.data import Data


def append_node_indices(dataframe):
    dataframe['node_index'] = dataframe.groupby('fname').cumcount()
    # import ipdb; ipdb.set_trace()
    return dataframe

def append_graph_indices(dataframe):
    fname_list = dataframe['fname'].values.tolist()
    fname_set = set(fname_list)
    fname_indices = [(fname, index) for index, fname in enumerate(fname_set)]
    fname_indices_df = pd.DataFrame(fname_indices, columns=['fname', 'fname_index'])
    dataframe = pd.merge(dataframe, fname_indices_df, how='left')
    return dataframe

def append_available_labels(dataframe, labels_dataframe):
    dataframe = pd.merge(dataframe, labels_dataframe, on=['fname'], how='left')
    return dataframe

def all_combination(list1, list2):
    combs = []
    for i in list1:
        for j in list2:
            if i == j:
                continue
            combs.append((i, j))
    if combs != []:
        return combs
    return []

def create_edge_index(dataframe, fname_set):

    edge_index = []
    for fname, _ in fname_set:
        fname_node_df = dataframe[dataframe.fname == fname]['node_index']
        fname_node_indices = fname_node_df.values.tolist()
        edge_index.append(all_combination(fname_node_indices, fname_node_indices))
    edge_index_slices = [len(ele) for ele in edge_index]
    edge_index_slices.insert(0, 0)

    edge_index = [item for sublist in edge_index for item in sublist]

    return edge_index, edge_index_slices
    
def save_data_to_pt(dataframe):
    fname_set = dataframe[['fname', 'fname_index']].drop_duplicates().values.tolist()
    fname_set = sorted(fname_set, key=lambda x: x[1])

    edge_index, edge_index_slices = create_edge_index(dataframe, fname_set)
    edge_index_slices = torch.tensor(edge_index_slices)
    import ipdb; ipdb.set_trace()
    edge_index_slices = torch.cumsum(edge_index_slices, dim=0)

    edge_index = torch.LongTensor(edge_index).t().contiguous()

    fname_counts = Counter(dataframe['fname'].values.tolist())

    x_slices = [fname_counts[fname] for fname, _ in fname_set]
    x_slices.insert(0, 0)
    x_slices = torch.tensor(x_slices)
    x_slices = torch.cumsum(x_slices, dim=0)
    x = dataframe[['image_height', 'image_width', 'start_x', 'start_y', 'end_x', 'end_y', 'mask_x']].values
    x = torch.from_numpy(x)

    fname_labels = dataframe[['fname', 'distancing']].drop_duplicates().values.tolist()
    fname_labels = {item[0]: int(item[1]) for item in fname_labels}
    y = [fname_labels[fname] for fname, _ in fname_set]
    y = torch.tensor(y)
    y_slices = [i for _, i in fname_set]
    y_slices.append(2872)
    y_slices = torch.tensor(y_slices)


    data = Data(x=x, edge_index=edge_index, y=y)
    dataset = (data, {'edge_index': edge_index_slices, 'x': x_slices, 'y': y_slices})
    import ipdb ; ipdb.set_trace()


            
    torch.save(tuple(dataset), 'test.pt')





if __name__ == "__main__":
    csv_path = "/home/santapo/OnlineLab/challenges/5k_compliance_zalo/distancing_compliance_gnn/data/gnn_data.csv"
    distancing = pd.read_csv("/home/santapo/OnlineLab/challenges/5k_compliance_zalo/distancing_compliance_gnn/data/train_meta.csv")
    distancing = distancing[['fname', 'distancing', 'mask', '5k']]
    dataframe = pd.read_csv(csv_path)

    dataframe = append_available_labels(dataframe, distancing)
    dataframe = dataframe.dropna(subset=['distancing']).reset_index()
    dataframe = append_graph_indices(dataframe).sort_values(by=['fname_index'])
    dataframe = append_node_indices(dataframe)

    save_data_to_pt(dataframe)
    


