import os
import torch
import xml.etree.ElementTree as ET
from os.path import exists

import pandas


def read_thyme_dataset(test_split=False):
    if test_split:
        text_folder = '/media/timotej/thyme_decrypted/THYME_2017_year/THYME/text_files/thyme-corpus/test'
    else:
        text_folder = '/media/timotej/thyme_decrypted/THYME_2017_year/THYME/text_files/thyme-corpus/train'
    text_files = os.listdir(text_folder)
    if test_split:
        annotation_folder = '/media/timotej/thyme_decrypted/THYME_2017_year/THYME/goldstandard_temporal/thyme-data-unzipped/coloncancer/Test'
    else:
        annotation_folder = '/media/timotej/thyme_decrypted/THYME_2017_year/THYME/goldstandard_temporal/thyme-data-unzipped/coloncancer/Train'
    annotation_file_folders = os.listdir(annotation_folder)

    all_relations = []

    for file in annotation_file_folders:
        if file.startswith("."):
            continue
        open_file = open(text_folder + "/" + file, "r")
        text = open_file.read()
        open_file.close()

        file_path = annotation_folder + "/" + file + "/" + file + ".Temporal-Relation.gold.completed.xml"
        if not exists(file_path):
            continue
        open_file = open(file_path, "r")
        annotation_text = open_file.read()
        open_file.close()
        root = ET.fromstring(annotation_text)
        events = {}
        relations = []
        for child in root:
            if child.tag == "annotations":
                for entity in child:
                    if entity.tag == "entity":
                        id = None
                        span = None
                        type = None
                        parentsType = None
                        event_text = ""
                        for p in entity:
                            if p.tag == "id":
                                id = p.text
                            if p.tag == "span":
                                span = [int(a) for a in p.text.split(";")[0].split(",")]
                                event_text = text[span[0]:span[1]]
                            if p.tag == "type":
                                type = p.text
                            if p.tag == "parentsType":
                                parentsType = p.text
                        events[id] = {"id": id,
                                      "span": span,
                                      "type": type,
                                      "parentsType": parentsType,
                                      "eventText": event_text}
                    elif entity.tag == "relation":
                        id = None
                        type = None
                        parentsType = None
                        Source = None
                        Target = None
                        relation = None
                        for p in entity:
                            if p.tag == "id":
                                id = p.text
                            if p.tag == "type":
                                type = p.text
                            if p.tag == "parentsType":
                                parentsType = p.text
                            if p.tag == "properties":
                                for p2 in p:
                                    if p2.tag == "Source":
                                        Source = p2.text
                                    if p2.tag == "Target":
                                        Target = p2.text
                                    if p2.tag == "Type":
                                        relation = p2.text
                        relations.append({"id": id,
                                      "type": type,
                                      "parentsType": parentsType,
                                      "eventText": event_text,
                                      "source": events[Source],
                                      "target": events[Target],
                                      "relation": relation})
        pass
        for r in relations:
            all_relations.append([text, r['relation'], r['source']['span'][0], r['source']['span'][1], r['source']['type'],
                              r['target']['span'][0], r['target']['span'][1], r['target']['type'],
                              r['source']['eventText'], r['target']['eventText'],
                              file, 'thyme'])
    df = pandas.DataFrame(all_relations)
    df.columns = ['text', 'class', 'event1_start', 'event1_end', 'event1_type', 'event2_start', 'event2_end',
                  'event2_type', 'event1_text',
                  'event2_text', 'document_id', 'source']
    return df

if __name__ == '__main__':
    import cryptpandas as crp
    df = read_thyme_dataset(test_split=False)
    crp.to_encrypted(df, password=os.environ['TP'], path='thyme.crypt')
    df = read_thyme_dataset(test_split=True)
    crp.to_encrypted(df, password=os.environ['TP'], path='thyme_test.crypt')
    print(df)
