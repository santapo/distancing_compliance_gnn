import pandas as pd


def append_node_indices(dataframe):
    num_nodes = dataframe.shape[0]
    node_indices = [i for i in range(1, num_nodes+1)]
    dataframe['node_index'] = node_indices
    return dataframe

def append_graph_indices(dataframe):
    fname_list = dataframe['fname'].values().tolist()
    fname_indices = [(fname, index) for index, fname in enumerate(fname_list)]
    fname_indices_df = pd.DataFrame(fname_indices, columns=['fname', 'fname_index'])
    dataframe = pd.merge(dataframe, fname_indices_df, how='left')
    return dataframe

def append_available_labels(dataframe, labels_dataframe):
    dataframe = pd.merge(dataframe, labels_dataframe)
    return dataframe

def all_combination(list1, list2):
    combs = []
    for i in list1:
        for j in list2:
            if i == j:
                continue
            combs.append((i, j))
    return combs

def create_adjacency_matrix(dataframe):
    fname_list = dataframe['fname'].values().tolist()
    
    edge_index = []
    for fname in fname_list:
        fname_node_df = dataframe[dataframe.fname == fname]
        fname_node_indices = fname_node_df.values().tolist()
        edge_index.append(
            all_combination(fname_node_indices, fname_node_indices))
    
    return edge_index



# def create_graph_indicator(dataframe):
    

if __name__ == "__main__":
    csv_path = ""
    # dataframe = pd.read_csv(csv_path)
    edge_index = create_edge_index(csv_path)
