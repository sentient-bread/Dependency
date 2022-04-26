from edgescorer import EdgeScorer 
from pos import POSTag
from copy import deepcopy
from math import inf

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
        # Path keeps follows the node upwards
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
    the highest-scored incoming edge to that cycle.
    (NOT to the component)
    """
    cycle, not_in_cycle = component
    component = cycle + not_in_cycle
    # cycle = list of nodes actually part of a cycle
    # Rest of the nodes in component are children

    for i in range(len(scores)):
        max_incoming = max(scores[i])
        scores[i] = list(map(lambda x: x-max_incoming, scores[i]))
        # Normalise scores (without affecting get_MST()'s copy)

    incoming_edges = []
    for node in cycle:
        incoming_edges += [(0, node)] + \
                          [(i, node) for i in range(1, len(scores))
                                     if (i != node and i not in component)]
        # Candidate incoming edges are those external
        # to the component
        # (src, trg)
    
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
        # an edge external to the component

    return mst

class Parser():
    def __init__(self, from_pretrained=True):
        if not from_pretrained:
            train_POS("../data/UD_English-Atis/en_atis-ud-train.conllu", 20)
            train_edgescorer("../data/UD_English-Atis/en_atis-ud-train.conllu", 35)

        # After training, the models are saved
        # If from_pretrained is True, then it is assumed
        # that the saved models use the same vocabulary
        # and words_to_indices
        self.pos_tagger = torch.load("../models/pos_model.pth")
        self.edgescorer = torch.load("../models/edgescorer_model.pth")

    def create_graph(self, sentence):
        """
        Accepts a list of indices
        including <BOS> but not <EOS>
        """
        scores = self.edgescorer(sentence)
        # Now scores[i][j] = probability that
        # i is the head of j

        pad_index = len(self.edgescorer.vocab) - 2
        # The number representing <PAD> in the sentence

        try:
            remove_index = sentence.tolist().index(pad_index)
            # The index in the sentence from which padding
            # has happened .: which has to be ignored
            scores = scores[:remove_index, :remove_index]
            # Stripping scores matrix to remove pad-pad dependencies
        except ValueError: pass
        # If the sentence has no pad_index

        scores = scores.transpose(0,1).tolist()
        # Now scores[i][j] = probability that
        # j is a head of i

        graph = [ [] ]*(len(sentence))
        # Initialised with [] as a placeholder

        for i in range(1, len(sentence)):
            graph[i] = [0] + [j for j in range(1, len(sentence)) if j != i]
        # graph[i] = list of nodes j s.t. there is an edge j -> i

        MST = get_MST(graph, scores)
