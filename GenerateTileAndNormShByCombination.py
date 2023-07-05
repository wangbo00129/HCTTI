import os
from itertools import product
from collections import OrderedDict

def generateNewFileAccordingToTemplateAndDict(path_template, dict_to_replace, path_output):
    str_all_lines = ''.join(open(path_template))
    for k, v in dict_to_replace.items():
        str_all_lines = str_all_lines.replace(str(k), str(v))
    open(path_output, 'w').write(str_all_lines)

def iterateOverDictsToGenerateListOfDictsOfSingleCombinations(dictionary):
    '''
    Input dictionary is like:
    {'A':[1,2], 'B':[3,4]}
    output will be like:
    [{'A':1, 'B':3}, {'A':1, 'B':4}, {'A':2, 'B':3}, {'A':2, 'B':4}]
    '''
    dictionary = OrderedDict(dictionary)
    output_dicts = []
    ks = dictionary.keys()
    for vs in product(*dictionary.values()):
        output_dicts.append(dict(zip(ks,vs)))
    return output_dicts

def generateAllCombinations():
    dict_all_possible_values = OrderedDict(
        [
            ('{env}', ['mpipytorch_env'],),
            ('{WORLD_SIZE}', [1,],),
            ('{str_normalizer}', ['NullNormalizer','TiatoolboxMacenkoNormalizer','TorchstainMacenkoNormalizer','MacenkoNormalizer','NumpyMacenkoNormalizer','ReinhardNormalizer','RuifrokNormalizer'],), 
            ('{str_writer_after_norm}', ['WriterInPng', 'WriterInHdf5'],),
            ('{use_cucim}', ['True', 'False'],),
            ('{read_region_device}', ['cpu'],),
            ('{read_region_num_workers}', [1,2,4]),
            ('{discard_page_cache}', ['false', 'true']),
        ]
    )
    dir_output = 'possible_sh'
    os.makedirs(dir_output, exist_ok=True)
    list_of_dicts = iterateOverDictsToGenerateListOfDictsOfSingleCombinations(dict_all_possible_values)
    for d in list_of_dicts:
        path_output = ','.join(['{}={}'.format(k,v) for k,v in d.items()]) + '.sh'
        path_output = path_output.replace('{', '').replace('}', '')
        path_output = os.path.join(dir_output, path_output)
        print(path_output)
        generateNewFileAccordingToTemplateAndDict(
            '/home/wangb/pipelines/changablepipelineforsvspreprocessor/TileAndNormSh.template', 
            d, 
            path_output
            )

if __name__ == '__main__':
    generateAllCombinations()
