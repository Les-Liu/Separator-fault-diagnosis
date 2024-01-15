import numpy as np
import pandas as pd
import itertools

def Sort_By_Value(dict:dict):
    """
    Using the values in the dictionary, sort from smallest to largest
    :param dict: Dictionaries to be processed
    :return: Outcome of the process
    """
    sort_list = sorted(dict.items(),key=lambda x : x[1])
    sort_dict = {}
    for n, s in sort_list:
        sort_dict[n] = s 
    return sort_dict

def Noisy_MAX(data:pd.DataFrame):
    """
    Completion of node CPT using Noisy-MAX algorithm
    :param file_path: File path
    :param file_name: File name
    :return: CPT for all of the nodes
    """
    # Get the state of this node
    state_num = int(data["state_num"].dropna())
    data.drop(["state_num"],axis=1,inplace=True)
    node_states = list(data.columns[-state_num:])
    
    # Get the parent of this node
    parent_nodes = list(data.columns[:int(len(list(data.columns[:-state_num]))/2)])
    
    # Count the states of each parent node and assign each state of the parent node a corresponding number
    parent_nodes_states = []
    states_number = {}
    length = []
    for i in range(len(parent_nodes)):
        states_number[parent_nodes[i]] = Sort_By_Value(dict(zip(list(data[parent_nodes[i]].unique()),
                                                                list(data[parent_nodes[i]+"_number"].unique()))))
        part_parent_nodes_states = []
        for j in states_number[parent_nodes[i]]:
            part_parent_nodes_states.append(j)
        parent_nodes_states.insert(i,part_parent_nodes_states)
        length.insert(i,len(part_parent_nodes_states))

    length = np.prod(length)

    # Construct the conditions of the conditional probability table and convert them to the corresponding numerical numbers
    condition = list(itertools.product(*parent_nodes_states))
    condition_matrix = []
    for i in range(len(condition)):
        part_conditon = list(condition[i])
        for j in range(len(parent_nodes)):
            part_conditon[j] = states_number[parent_nodes[j]][part_conditon[j]]
        condition_matrix.insert(i,part_conditon)
    condition_matrix = np.array(condition_matrix)

    # Calculate the probability value of the node state
    basic_probability_condition = []
    for i in range(len(parent_nodes)):
         basic_probability_condition.insert(i,np.array(data[parent_nodes[i]+"_number"]))
    basic_probability_condition = np.array(basic_probability_condition).transpose()
    
    basic_probability = []
    for i in range(len(node_states)):
        basic_probability.insert(i,np.array(data[node_states[i]]))
    basic_probability = np.array(basic_probability).transpose()
    
    basic_probability_dict = {}
    for i in range(basic_probability_condition.shape[1]):
        part_basic_probability_dict = {}
        for j in range(basic_probability_condition.shape[0]):
            if basic_probability_condition[j,i] != 0:
                part_basic_probability_dict[basic_probability_condition[j,i]] = basic_probability[j,:]
            else:
                continue
        basic_probability_dict[i] = part_basic_probability_dict

    probability_matrix = np.zeros((length,len(node_states)))
    for i in range(probability_matrix.shape[0]):
        if i == 0:
            for j in range(probability_matrix.shape[1]):
                if j == 0:
                    probability_matrix[i,j] = 1
                else:
                    probability_matrix[i,j] = 0
        else:
            for j in range(probability_matrix.shape[1]):
                part_probability = [] 
                if j == 0:
                    for m in range(len(condition_matrix[i,:])):
                        if condition_matrix[i,m] != 0:
                            probability = basic_probability_dict[m][condition_matrix[i,m]][j]
                            part_probability.insert(m,probability)
                        else:
                            continue
                    probability_matrix[i,j] = np.prod(np.array(part_probability))
                elif (j > 0) and (j < probability_matrix.shape[1]-1):
                    sum_probability_list = []
                    for m in range(condition_matrix.shape[1]):
                        if condition_matrix[i,m] != 0:
                            sum_probability = sum(basic_probability_dict[m][condition_matrix[i,m]][:j+1])
                            sum_probability_list.insert(m,sum_probability)
                        else:
                            continue
                    probability_matrix[i,j] = np.prod(sum_probability_list)-sum(probability_matrix[i,0:j])
                elif j == probability_matrix.shape[1]-1:
                    probability_matrix[i,j] = 1-sum(probability_matrix[i,:][:j])

    final_conditon = []
    for i in range(len(condition)):
        part_conditon = list(condition[i])
        final_conditon.insert(i,part_conditon)
    final_conditon = np.array(final_conditon)

    CPT = np.concatenate((final_conditon,np.around(probability_matrix,decimals=8)),axis=1)
    final_CPT = pd.DataFrame(data=CPT,columns=parent_nodes+node_states)
    
    return final_CPT