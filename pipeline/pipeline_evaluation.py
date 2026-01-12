from pipeline.pipeline import *

def evaluate():
    from training.train_and_evaluate_relation_extraction import load_stored_dataset_combination_graph
    from custom_datasets.dataframe_dataset import DFDataset

    _, _, dataset_test_ub = load_stored_dataset_combination_graph(balanced=True, dataset="i2b2")
    number_of_relations = 3

    import nltk
    nltk.download('punkt_tab')
    dataframe = dataset_test_ub.df
    documents = set(dataframe["document_id"])
    dataset = []
    for doc in documents:
        relations = dataframe[dataframe["document_id"] == doc]
        text = relations["text"].iloc[0]
        graph = []
        for row in relations.iloc:
            graph.append(((row["event1_start"], row["event1_end"], row["event1_text"]), row["class"],
                          (row["event2_start"], row["event2_end"], row["event2_text"])))
        dataset.append({"text": text, "graph": graph})
    patient_ind = 1
    example = dataset[patient_ind]

    def overlap(event1, event2):
        start1, end1, text1 = event1
        start2, end2, text2 = event2
        if start1 >= start2 and start1 <= end2:
            # print(event1, event2)
            return True
        if end1 >= start2 and end1 <= end2:
            # print(event1, event2)
            return True
        if start2 >= start1 and start2 <= end1:
            # print(event1, event2)
            return True
        return False

    def most_simmilar_event(event_list, event_target):
        # min_difference = -1
        # best_match = None
        for e in event_list:
            if overlap(e, event_target):
                return e
            # dif = -1
            # if e in event_target:
            #     dif = len(event_target) - len(e)
            # if event_target in e:
            #     dif = len(e) - len(event_target)
            # if dif >= 0 and (min_difference < 0 or min_difference > dif):
            #     min_difference = dif
            #     best_match = e
        return None

    def find_relation(graph, event1, event2, expected_relation):
        for ind, r in enumerate(graph):
            if r[0] == event1 and r[2] == event2 and r[1] == expected_relation:
                return ind, r
        for ind, r in enumerate(graph):
            if r[0] == event1 and r[2] == event2:
                return ind, r
        return -1, None

    # compare_graphs(truth, prediction)
    def compare_graphs(graph1, graph2):
        event_map = {}
        events1 = list(set([r[0] for r in graph1] + [r[2] for r in graph1]))
        events2 = list(set([r[0] for r in graph2] + [r[2] for r in graph2]))
        for e in events1:
            event_map[e] = most_simmilar_event(events2, e)
        print(event_map)

        matching_relation = [False for _ in range(len(graph2))]

        correct = 0
        incorrect = 0
        missing = 0
        too_much = 0
        for relation in graph1:
            ind2, relation2 = find_relation(graph2, event_map[relation[0]], event_map[relation[2]], relation[1])
            if relation2 is None:
                missing += 1
            else:
                matching_relation[ind2] = True
                if relation[1] == relation2[1]:
                    correct += 1
                else:
                    incorrect += 1
        too_much = len(matching_relation) - sum(matching_relation)
        print(correct, incorrect, missing, too_much)
        return correct, incorrect, missing, too_much

    def compute_f1(our_graph, gold_graph):
        our_graph = [x for x in our_graph]
        gold_graph = [x for x in gold_graph]
        our_graph_closure = graph_closure([x for x in our_graph])
        gold_graph_closure = graph_closure([x for x in gold_graph])
        system_gold_plus, _, _, _ = compare_graphs(gold_graph_closure, our_graph)
        precision = system_gold_plus / len(our_graph)
        system_plus_gold, _, _, _ = compare_graphs(gold_graph, our_graph_closure)
        recall = system_plus_gold / len(gold_graph)
        f1 = 2*(precision*recall)/(precision+recall)
        return precision, recall, f1, system_gold_plus, system_plus_gold, len(our_graph), len(gold_graph)

    def get_event_pairs_of_interest(example):
        graph = example["graph"]
        event_pairs = []
        for g in graph:
            event_pairs.append((g[0], g[2]))
        return list(event_pairs)


    def analyze_document(text, patient_id, event_pairs_of_interest=None):
        events = extract_events(text)
        # print("Events:")
        # print(events)
        if event_pairs_of_interest is None:
            event_pairs = generate_event_pairs(text, events)
        else:
            event_pairs = []
            for e1, e2 in event_pairs_of_interest:
                e1 = most_simmilar_event(events, e1)
                e2 = most_simmilar_event(events, e2)
                if e1 is not None and e2 is not None:
                    event_pairs.append((e1, e2))
        # print("Pairs ("+str(len(event_pairs))+"):")
        # print(event_pairs)
        dataframe = construct_basic_dataframe(text, event_pairs, 0)
        # print("Dataframe")
        # print(dataframe)
        dataset = construct_dataset_with_graphs(text, dataframe, patient_id)
        # Window text into the same format as when training the model (windowing is already performed in convert_row_to_graph function
        # dataset.generated = list(filter(lambda x: x is not None, map(window_text, dataset.generated)))
        relations = predict_temporal_relations(dataset)
        return relations

    transitivity = {
        ("OVERLAP", "OVERLAP"): "OVERLAP",
        ("BEFORE", "OVERLAP"): "BEFORE",
        ("BEFORE", "BEFORE"): "BEFORE",
        ("OVERLAP", "BEFORE"): "BEFORE",
        ("AFTER", "OVERLAP"): "AFTER",
        ("OVERLAP", "AFTER"): "AFTER",
        ("AFTER", "AFTER"): "AFTER",
        ("AFTER", "BEFORE"): "OVERLAP",
        ("BEFORE", "AFTER"): "OVERLAP"
    }

    inverse = {
        "BEFORE": "AFTER",
        "AFTER": "BEFORE",
        "OVERLAP": "OVERLAP"
    }

    def does_relation_exist(graph, event1, event2):
        ind, relation = find_relation(graph, event1, event2, None)
        return ind >= 0

    def add_transitive(graph):
        for i in range(len(graph)):
            for j in range(i + 1, len(graph)):
                if graph[i][2] == graph[j][0]:
                    # matching relations
                    event1 = graph[i][0]
                    event2 = graph[j][2]
                    if not does_relation_exist(graph, event1, event2):
                        relation = transitivity[(graph[i][1], graph[j][1])]
                        new_relation = (event1, relation, event2)
                        graph.append(new_relation)
        return graph

    def add_inverse(graph):
        for i in range(len(graph)):
            event1 = graph[i][2]
            event2 = graph[i][0]
            if not does_relation_exist(graph, event1, event2):
                relation = inverse[graph[i][1]]
                new_relation = (event1, relation, event2)
                graph.append(new_relation)
        return graph

    def graph_closure(graph):
        graph = add_inverse(graph)
        graph = add_transitive(graph)
        return graph

    f = open("end-to-end-pipeline-results.txt", "a")
    for patient_ind, example in enumerate(dataset):
        print(str(patient_ind) + "/" + str(len(dataset)))
        event_pairs_of_interest = get_event_pairs_of_interest(example)
        graph_predicted = analyze_document(example["text"], patient_ind,
                                           event_pairs_of_interest=event_pairs_of_interest)
        graph_predicted = [(tuple(triplet[0]), triplet[1], tuple(triplet[2])) for triplet in graph_predicted]
        graph1 = example["graph"]
        graph2 = graph_predicted
        correct, incorrect, missing, too_much = compare_graphs(graph1, graph2)
        f.write(str(patient_ind))
        f.write(", ")

        f.write(str(correct))
        f.write(", ")
        f.write(str(incorrect))
        f.write(", ")
        f.write(str(missing))
        f.write(", ")
        f.write(str(too_much))
        f.write(", ")

        graph1_c = graph_closure(graph1)
        graph2_c = graph_closure(graph2)
        correct, incorrect, missing, too_much = compare_graphs(graph1_c, graph2_c)
        f.write(str(correct))
        f.write(", ")
        f.write(str(incorrect))
        f.write(", ")
        f.write(str(missing))
        f.write(", ")
        f.write(str(too_much))

        # p, r, f1, system_gold_plus, system_plus_gold, our_len, gold_len
        system_gold_plus, _, _, _ = compare_graphs(graph1_c, graph2)
        system_plus_gold, _, _, _ = compare_graphs(graph1, graph2_c)
        our_len = len(graph2)
        gold_len = len(graph1)
        p = system_gold_plus / our_len
        r = system_plus_gold / gold_len
        f1 = 2 * p * r / (p + r)

        f.write(", ")
        f.write(str(p))
        f.write(", ")
        f.write(str(r))
        f.write(", ")
        f.write(str(f1))
        f.write(", ")
        f.write(str(system_gold_plus))
        f.write(", ")
        f.write(str(system_plus_gold))
        f.write(", ")
        f.write(str(our_len))
        f.write(", ")
        f.write(str(gold_len))

        f.write("\n")
        f.flush()
    f.close()

if __name__ == '__main__':
    evaluate()