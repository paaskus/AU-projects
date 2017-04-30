import pandas as pd
import json

jp_data = pd.read_csv('~/JPdatatool/JPdata/jyllandsposten_20170402-20170402_18014v2.tsv',
                   delimiter = '\t',
                   error_bad_lines = False,
                   low_memory = False,
                   index_col = 0
                )
                
only_sso_id = jp_data.loc[(jp_data['sso'] != 'NOTSET') & (pd.notnull(jp_data['sso']))]

grouped_by_sso_id = only_sso_id.groupby(['sso', 'Platform']).size().to_frame(name = 'Count').reset_index()

grouped_by_sso_id_max = grouped_by_sso_id.groupby(['sso'], sort = False)['Platform','Count'].max()

# RESULTS
total_platform_count = jp_data['Platform'].value_counts()

sso_id_platform_count = only_sso_id['Platform'].value_counts()

primary_platform_counts = grouped_by_sso_id_max['Platform'].value_counts()

# RESULTS AS JSON

total_platform_count_json = json.loads(total_platform_count.to_json())

sso_id_platform_count_json = json.loads(sso_id_platform_count.to_json())

primary_platform_counts_json = json.loads(primary_platform_counts.to_json())

combined_platform_result = {'total': total_platform_count_json,
                            'sso': sso_id_platform_count_json,
                            'primary': primary_platform_counts_json}
