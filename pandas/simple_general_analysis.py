import pandas as pd
import json


def filter_data(data, filter_options_string):
    filter_options = json.loads(filter_options_string)
    filters = filter_options['filters']
    result = data
    for f in filters:
        filter_name = f['filterName']
        selected = f['selected']
        result = result[result[filter_name].isin(selected)]
    return result

def simple_analysis(jp_data, attribute_name):    
    # Select observations where SSO-ID is set
    only_sso_id = jp_data.loc[(jp_data['sso'] != 'NOTSET') & (pd.notnull(jp_data['sso']))]
    
    sso_id_count = pd.DataFrame({'No result': 1}, index = ['No result'])
    primary_counts = sso_id_count
    
    if not only_sso_id.empty:
        # Find primary ...
        grouped_by_sso_id = only_sso_id.groupby(['sso', attribute_name]).size().to_frame(name = 'Count').reset_index()
        grouped_by_sso_id_max = grouped_by_sso_id.groupby(['sso'], sort = False)[attribute_name, 'Count'].max()
        
        # Results        
        sso_id_count = only_sso_id[attribute_name].value_counts()
        primary_counts = grouped_by_sso_id_max[attribute_name].value_counts()

    # Results
    total_count = jp_data[attribute_name].value_counts()

    # Results as JSON
    total_count_json = json.loads(total_count.to_json())
    sso_id_count_json = json.loads(sso_id_count.to_json())
    primary_counts_json = json.loads(primary_counts.to_json())

    # Merge the three separate JSON objects to a combined JSON object
    combined_result = {'total': total_count_json,
                       'sso': sso_id_count_json,
                       'primary': primary_counts_json}
    
    return combined_result

def crunch_the_data(path_to_data, filter_options):
    data = pd.read_csv(path_to_data,
                   delimiter = '\t',
                   error_bad_lines = False,
                   low_memory = False,
                   index_col = 0
                )
                
    filtered_data = filter_data(data, filter_options)
    
    result = {}
    list_of_attributes = ['Platform', 'Device', 'Operating system', 'Web browser', 'Category']
    
    for a in list_of_attributes:
        result[a] = simple_analysis(filtered_data, a)
        
    return json.dumps(result)
    
def test():
    filter_options = '{"filters":[]}'
    result = crunch_the_data('~/JPdatatool/JPdata/jyllandsposten_20170402-20170402_18014v2.tsv', filter_options)
    print(result)
    return result