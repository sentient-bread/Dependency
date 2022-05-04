from edgescorer import EdgeScorer 
from pos import POSTag
from copy import deepcopy
from math import inf
import torch

def get_cycles(mst):
    """
    Takes in a tentative MST and returns a list
    of all components containing cycles.
    """
    cycles = []
    for node in range(len(mst)):
        already_seen = any(node in cycle for cycle in cycles)
        if (already_seen): continue
        # If the node is part of a previously encountered cycle, ignore

        is_cycle = 0
        curr_node = mst[node]
        path = [node]
        # Path follows the node upwards
        while (curr_node != []):
            if (curr_node == node):
                # Node is part of a cycle
                is_cycle = 1
                break

            if (curr_node in path):
                # node leads to a cycle but is not part of one
                # which means the while loop would be inf
                break

            path.append(curr_node)
            curr_node = mst[curr_node]
            # Follow path up

        if (is_cycle == 1):
            cycles.append(path)

    # Now we need the cycles to include the
    # entire connected component
    cyclic_components = []
    for cycle in cycles:
        component = cycle.copy()
        outside_component_deps = list(filter(lambda j: j not in component
                                                   and mst[j] in component,
                                   range(len(mst))))

        component_minus_cycle = outside_component_deps
        component += outside_component_deps

        while (outside_component_deps != []):                                   # As long as there are nodes outside the component
            outside_component_deps = list(filter(lambda j: j not in component   # and dependent on it,
                                                       and mst[j] in component, # add them to the component
                                                 range(len(mst))))

            component_minus_cycle += outside_component_deps
            component += outside_component_deps
        
        cyclic_components.append((cycle, component_minus_cycle))

    return cyclic_components

def best_incoming_to_cycle(component, scores):
    """
    Given a component containing a cycle, finds
    the highest-scored incoming edge to that cycle
    (NOT to the component).
    """
    cycle, not_in_cycle = component
    component = cycle + not_in_cycle
    # cycle = list of nodes actually part of a cycle
    # Rest of the nodes in component are children/
    # descendants of nodes in the cycle.

    for i in range(len(scores)):
        max_incoming = max(scores[i])
        scores[i] = list(map(lambda x: x-max_incoming, scores[i]))
        # All nodes incoming to the cycle have to be compared,
        # but they have different destinations. We normalise
        # by subtracting the maximums .: all edges part of the cycle
        # have weight zero now.
        # This is done without affecting get_MST()'s copy of scores.

    incoming_edges = []
    for node in cycle:
        incoming_edges += [(0, node)] + \
                          [(i, node) for i in range(1, len(scores))
                                     if (i != node and i not in component)]
        # Candidate incoming edges are those external
        # to the component.
        # Format is (src, trg).
    
    best_edge = max(incoming_edges, key = lambda edge: scores[edge[1]][edge[0]])
    # Edge with max score

    return best_edge

def get_MST(graph, scores):
    """
    Given a graph and a score matrix, finds
    the MST.
    """
    mst = [[]]*(len(graph))
    for i in range(1, len(graph)):
        mst[i] = max(graph[i], key = lambda j: scores[i][j])
    # mst[i] = parent of i

    cyclic_components = get_cycles(deepcopy(mst))
    # cyclic_components = [(cycle, nodes_external_to_cycle)]
    # These are connected components

    for component in cyclic_components:
        edge = best_incoming_to_cycle(component, deepcopy(scores))
        src, trg = edge
        # Find the best incoming edge to the cycle (NOT the component)
        mst[trg] = src
        # Replace one edge inside the cycle with
        # an edge external to the component. The cycle
        # is now eliminated

    return mst

def create_graph(parser, tokens):
    """
    Accepts a list of indices
    including <BOS> but not <EOS>
    """
    indices = torch.tensor([parser.index(tok) for tok in tokens])
    pos = torch.tensor([0 for _ in indices])
    heads = torch.tensor([0 for _ in indices])
    labels = torch.tensor([0 for _ in indices])

    word_level = torch.stack([indices, pos, heads, labels], dim=0).unsqueeze(0)

    max_word_len = max([len(tok) for tok in tokens])
    char_level = []
    for tok in tokens:
      idx = [parser.character_level_model.character_to_indices[c] if c in parser.character_level_model.character_to_indices.keys() else len(parser.character_level_model.character_vocab)-2 for c in tok]
      idx = [len(parser.character_level_model.character_vocab)-2] * (max_word_len-len(tok)) + idx
    
      char_level.append(idx)
    
    char_level = torch.tensor(char_level).unsqueeze(0)

    scores, labels = parser([word_level, char_level])
    tags = parser.pos_tagger(indices)
    # Now scores[i][j] = probability that
    # i is the head of j

    #pad_index = len(self.edgescorer.vocab) - 2
    # The number representing <PAD> in the sentence

    #try:
    #    remove_index = sentence.tolist().index(pad_index)
    #    # The index in the sentence from which padding
    #    # has happened .: which has to be ignored
    #    scores = scores[:remove_index, :remove_index]
    #    # Stripping scores matrix to remove pad-pad dependencies
    #except ValueError: pass
    # If the sentence has no pad_index

    scores = scores[0]
    labels = labels[0]
    scores = scores.tolist()
    # Now scores[i][j] = probability that
    # j is a head of i

    graph = [ [] ]*(len(tokens))
    # Initialised with [] as a placeholder

    for i in range(1, len(tokens)):
        graph[i] = [0] + [j for j in range(1, len(tokens)) if j != i]
    # graph[i] = list of nodes j s.t. there is an edge j -> i

    MST = get_MST(graph, scores)
    
    return MST, labels.argmax(dim=1), tags.argmax(dim=1)
