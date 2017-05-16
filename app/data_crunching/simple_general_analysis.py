# takes a DataFrame and discards all empty rows (where all attributes are empty)
def __discard_empty_rows(data):
    return data.dropna(axis=0, how='all')

# takes a DataFrame and discards all rows where SSO ID is not set
def __select_sso_data(data):
    return data.loc[(data['sso'] != 'NOTSET')]

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
    
    total_count_json = no_result_json
    sso_id_count_json = no_result_json
    primary_count_json = no_result_json
    
    # Merge the three separate JSON objects to a combined JSON object
    combined_result = {'total': total_count_json,
                       'sso': sso_id_count_json,
                       'primary': primary_count_json}
    
    # total count
    if data.empty: return combined_result
    else: combined_result['total'] = json.loads(
        clean_data[attribute_name].value_counts().to_json()
    )
    
    # select rows where SSO ID is set
    only_sso_id_data = __select_sso_data(clean_data)
    
    # SSO ID count
    if only_sso_id_data.empty: return combined_result
    else: combined_result['sso'] = json.loads(
        only_sso_id_data[attribute_name].value_counts().to_json()
    )
    
    # primary count
    grouped_by_sso_id = only_sso_id_data.groupby(['sso', attribute_name]).size().to_frame(name = 'Count').reset_index()
    grouped_by_sso_id_max = grouped_by_sso_id.groupby(['sso'], sort = False)[attribute_name, 'Count'].max()
    combined_result['primary'] = json.loads(
        grouped_by_sso_id_max[attribute_name].value_counts().to_json()
    )
    
    return combined_result

def crunch_the_data(path_to_data, filter_options):
    import pandas as pd
    import json
    data = pd.read_csv(path_to_data,
                       delimiter = '\t',
                       error_bad_lines = False,
                       low_memory = False,
                       index_col = 0
                       )
                
    filtered_data = filter_data(data, filter_options)
    list_of_attributes = ['Platform', 'Device', 'Operating system', 'Web browser', 'Category']
    
    result = {}
    
    for a in list_of_attributes:
        result[a] = simple_analysis(filtered_data, a)
        
    return json.dumps(result)
    
def test():
    filter_options = '{"filters":[]}'
    result = crunch_the_data('~/JPdatatool/JPdata/jyllandsposten_20170402-20170402_18014v2.tsv', filter_options)
    print(result)
    return result