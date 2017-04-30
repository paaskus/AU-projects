import pandas as pd
import json

def simple_analysis(path_to_data, attribute_name):
    jp_data = pd.read_csv(path_to_data,
                   delimiter = '\t',
                   error_bad_lines = False,
                   low_memory = False,
                   index_col = 0
                )
    
    # Select observations where SSO-ID is set
    only_sso_id = jp_data.loc[(jp_data['sso'] != 'NOTSET') & (pd.notnull(jp_data['sso']))]
    
    # Find primary ...
    grouped_by_sso_id = only_sso_id.groupby(['sso', attribute_name]).size().to_frame(name = 'Count').reset_index()
    grouped_by_sso_id_max = grouped_by_sso_id.groupby(['sso'], sort = False)[attribute_name, 'Count'].max()

    # Results
    total_count = jp_data[attribute_name].value_counts()
    sso_id_count = only_sso_id[attribute_name].value_counts()
    primary_counts = grouped_by_sso_id_max[attribute_name].value_counts()

    # Results as JSON
    total_count_json = json.loads(total_count.to_json())
    sso_id_count_json = json.loads(sso_id_count.to_json())
    primary_counts_json = json.loads(primary_counts.to_json())

    # Merge the three separate JSON objects to a combined JSON object
    combined_result = {'total': total_count_json,
                       'sso': sso_id_count_json,
                       'primary': primary_counts_json}
    
    return combined_result

platform = simple_analysis('~/JPdatatool/JPdata/jyllandsposten_20170402-20170402_18014v2.tsv', 'Platform')
device = simple_analysis('~/JPdatatool/JPdata/jyllandsposten_20170402-20170402_18014v2.tsv', 'Device')
operating_system = simple_analysis('~/JPdatatool/JPdata/jyllandsposten_20170402-20170402_18014v2.tsv', 'Operating system')
web_browser = simple_analysis('~/JPdatatool/JPdata/jyllandsposten_20170402-20170402_18014v2.tsv', 'Web browser')
category = simple_analysis('~/JPdatatool/JPdata/jyllandsposten_20170402-20170402_18014v2.tsv', 'Category')
