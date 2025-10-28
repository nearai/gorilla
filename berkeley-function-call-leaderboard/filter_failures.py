import json

# Load and filter out objects with "traceback" key
filtered_data = []
file_name='/workspaces/nearaiml/eval/gorilla/berkeley-function-call-leaderboard/result/zai-org_GLM-4.5-FC/agentic/memory/kv/BFCL_v4_memory_kv_prereq_result.json'

with open(file_name, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            
            # Only keep if "traceback" key is NOT present
            if "traceback" not in data:
                filtered_data.append(data)

print(f"Kept {len(filtered_data)} objects (removed ones with traceback)")

# Now work with the filtered data
for data in filtered_data:
    print(f"ID: {data['id']}")
    # Process as normal...

#%%
import json
import os

model_result_path='/workspaces/nearaiml/eval/gorilla/berkeley-function-call-leaderboard/result/deepseek-ai_DeepSeek-V3.1-FC/'
#make a list of all the json files in the model_result_path
file_names= [
    model_result_path + "agentic/memory/kv/BFCL_v4_memory_kv_prereq_result.json",
    model_result_path + "agentic/memory/kv/BFCL_v4_memory_kv_result.json",
    model_result_path + "agentic/memory/rec_sum/BFCL_v4_memory_rec_sum_prereq_result.json",
    model_result_path + "agentic/memory/rec_sum/BFCL_v4_memory_rec_sum_result.json",
    model_result_path + "agentic/memory/vector/BFCL_v4_memory_vector_prereq_result.json",
    model_result_path + "agentic/memory/vector/BFCL_v4_memory_vector_result.json",
    model_result_path + "agentic/BFCL_v4_web_search_base_result.json",
    model_result_path + "agentic/BFCL_v4_web_search_no_snippet_result.json",
] + [model_result_path + "live/" + filename for filename in os.listdir(model_result_path + "live") if filename.endswith(".json")] + [model_result_path + "multi_turn/" + filename for filename in os.listdir(model_result_path + "multi_turn") if filename.endswith(".json")] + [model_result_path + "non_live/" + filename for filename in os.listdir(model_result_path+ "non_live") if filename.endswith(".json")]
    
for file_name in file_names:  
    filtered_data = []  
    # Read and filter
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if "traceback" not in data:
                    filtered_data.append(data)

    # Save filtered results to new file
    with open(file_name, 'w', encoding='utf-8') as f:
        for data in filtered_data:
            f.write(json.dumps(data) + '\n')

    print(f"Saved {len(filtered_data)} filtered objects to filtered_output.json")
# %%
