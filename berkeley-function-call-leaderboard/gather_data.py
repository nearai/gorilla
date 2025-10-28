#%%
import json

all_data = []
file_name='/workspaces/nearaiml/eval/gorilla/berkeley-function-call-leaderboard/score/zai-org_GLM-4.6-FC/agentic/BFCL_v4_web_search_no_snippet_score.json'

# Read and filter
with open(file_name, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            all_data.append(data)

# %%
import pandas as pd
import json

fixed_data=[]
for data in all_data[1:]:
    data['inference_log']=json.dumps(data['inference_log'])
    fixed_data.append(data)

df=pd.DataFrame(all_data[1:])
df.to_csv('/workspaces/nearaiml/eval/BFCL_v4_web_search_no_snippet_score.csv', index=False)
# %%
