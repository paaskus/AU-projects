# takes a DataFrame and discards all empty rows (where all attributes are empty)
def __discard_empty_rows(data):
    return data.dropna(axis=0, how='all')

# takes a DataFrame and discards all rows where SSO ID is not set
def __select_sso_data(data):
    return data.loc[(data['sso'] != 'NOTSET')]

def __select_mapped_sso_data(data):
    import  pandas as pd
    return data[pd.notnull(data['mapped_sso'])]

def filter_data(data, filter_options_string):
    import json

    filter_options = json.loads(filter_options_string)
    filters = filter_options['filters']
    result = data
    for f in filters:
        filter_name = f['filterName']
        selected = f['selected']
        result = result[result[filter_name].isin(selected)]
    
    # output result to .csv, if user wants the raw, filtered data
    result.to_csv('~/JPdatatool/app/static/filtered_result.csv')    
    
    return result

def simple_analysis(data, attribute_name):
    import json
    
    # discard empty rows
    clean_data = __discard_empty_rows(data)
    
    no_result_json = {'No result': 0}
    
    total_count_json = {'normal' : no_result_json,
                        'mapped' : no_result_json}
    sso_id_count_json = {'normal' : no_result_json,
                        'mapped' : no_result_json}
    primary_count_json = {'normal' : no_result_json,
                        'mapped' : no_result_json}
    
    # Merge the three separate JSON objects to a combined JSON object
    combined_result = {'total': total_count_json,
                       'sso': sso_id_count_json,
                       'primary': primary_count_json}
    
    # total count
    if data.empty: return combined_result
    else: 
        combined_result['total']['normal'] = json.loads(
                clean_data[attribute_name].value_counts().to_json()
            )       
        combined_result['total']['mapped'] =  json.loads(
                clean_data[attribute_name].value_counts().to_json()
            ) 
    
    # select rows where SSO ID is set
    only_sso_id_data = __select_sso_data(clean_data)
    only_mapped_sso_data = __select_mapped_sso_data(clean_data)
    
    # SSO ID count
    if only_sso_id_data.empty: return combined_result
    else: 
        combined_result['sso']['normal'] = json.loads(
                only_sso_id_data[attribute_name].value_counts().to_json()
            )
        combined_result['sso']['mapped'] = json.loads(
                only_mapped_sso_data[attribute_name].value_counts().to_json()
            )
    
    # primary count 
    grouped_by_sso_id = only_sso_id_data.groupby(['sso', attribute_name]).size().to_frame(name = 'Count').reset_index()
    grouped_by_sso_id_max = grouped_by_sso_id.groupby(['sso'], sort = False)[attribute_name, 'Count'].max()
    combined_result['primary']['normal'] = json.loads(
        grouped_by_sso_id_max[attribute_name].value_counts().to_json()
    )
    
    grouped_by_mapped_sso = only_sso_id_data.groupby(['mapped_sso', attribute_name]).size().to_frame(name = 'Count').reset_index()
    grouped_by_mapped_sso_id_max = grouped_by_mapped_sso.groupby(['mapped_sso'], sort = False)[attribute_name, 'Count'].max()
    combined_result['primary']['mapped'] = json.loads(
        grouped_by_mapped_sso_id_max[attribute_name].value_counts().to_json()
    )
    return combined_result

def make_mapping(data):
    import pandas as pd
    
    mapping = data
    mapping = mapping[['cookie', 'sso']]
    mapping = __discard_empty_rows(mapping)
    mapping = __select_sso_data(mapping)
    mapping = mapping.drop_duplicates(subset='cookie', keep='first', inplace=False)
    mapping = mapping.set_index('cookie')
    mapping = mapping.rename(columns={'cookie': 'cookie', 'sso': 'mapped_sso'})
    data = data.set_index('cookie')
    result = pd.concat([data, mapping], join='outer', axis=1, join_axes=[data.index])   
    
    return result
    
def barrunasonify(old_object):
    new_object ={}
    for a in old_object:
        new_sub_object = {}
        for b in old_object[a]:
            normal = old_object[a][b]['normal']
            mapped = old_object[a][b]['mapped']   
            new_sub_sub_sub_object = {}
        
            for c in {**normal, **mapped}:
                if (c in normal and c in mapped): 
                    new_sub_sub_sub_object[c] = {'normal':normal[c],
                                                 'mapped':mapped[c]}
                if (c in normal and c not in mapped): 
                    new_sub_sub_sub_object[c] = {'normal':normal[c],
                                                 'mapped': 0}
                if (c not in normal and c in mapped): 
                    new_sub_sub_sub_object[c] = {'normal':0,
                                                 'mapped':mapped[c]}
            new_sub_object[b]=new_sub_sub_sub_object
        new_object[a] = new_sub_object
    
    return new_object
def crunch_the_data(path_to_data, filter_options):
    import pandas as pd
    import json
    data = pd.read_csv(path_to_data,
                       delimiter = '\t',
                       error_bad_lines = False,
                       low_memory = False,
                       index_col = 0
                       )
                
    mapped_data = make_mapping(data)
    filtered_data = filter_data(mapped_data, filter_options)
    list_of_attributes = ['Platform', 'Device', 'Operating system', 'Web browser', 'Category']
    
    result = {}
    
    for a in list_of_attributes:
        result[a] = simple_analysis(filtered_data, a)
        
    result = barrunasonify(result)
    
    return json.dumps(result)
    

def test():
    filter_options = '{"filters":[]}'
    result = crunch_the_data('~/JPdatatool/JPdata/jyllandsposten_20170402-20170402_18014v2.tsv', filter_options)
    print(result)
    return result


