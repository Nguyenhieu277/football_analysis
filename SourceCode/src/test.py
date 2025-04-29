import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast # Import the ast module

# Helper function to safely parse string representations of lists/arrays
def parse_embedding(embedding_str):
    if pd.isna(embedding_str):
        return None
    try:
        # Safely evaluate the string as a Python literal (list)
        return ast.literal_eval(embedding_str)
    except (ValueError, SyntaxError):
        # Handle cases where the string is not a valid list representation
        return None

df1 = pd.read_csv(r'D:\work\football_analysis\embed_df.csv')
df2 = pd.read_csv(r'D:\work\football_analysis\embed_df2.csv')

# Convert string embeddings to lists of floats
df1['embedding'] = df1['embedding'].apply(parse_embedding)
df2['embedding'] = df2['embedding'].apply(parse_embedding)


matches = []
matched_names = []
eligible_embeddings_clean = []
eligible_players_clean = []
eligible_embeddings = []

minutes_col = 'minutes'
player_col = 'player'

for idx, emb in enumerate(df2['embedding']):
    # Check if emb is a list/array and not None before appending
    if emb is not None and isinstance(emb, list):
        eligible_embeddings_clean.append(emb)
        # Ensure player name is appended correctly based on the original index 'idx'
        eligible_players_clean.append(df2.loc[idx, 'player']) # Use df2.loc[idx, 'player']

# Convert the list of lists to a 2D NumPy array
# Ensure all elements in eligible_embeddings_clean are valid lists/arrays of the same dimension
# Filter out any None values that might have slipped through or resulted from parsing errors
eligible_embeddings_clean = [emb for emb in eligible_embeddings_clean if emb is not None]
eligible_players_clean = [player for emb, player in zip(eligible_embeddings_clean, eligible_players_clean) if emb is not None] # Keep players aligned

# Check if eligible_embeddings_clean is empty before converting to numpy array
if not eligible_embeddings_clean:
    print("Warning: No valid embeddings found in df2.")
    # Handle this case appropriately, maybe exit or skip the similarity calculation
    exit() # Or continue, depending on desired behavior

eligible_embeddings_clean = np.array(eligible_embeddings_clean, dtype=float) # Specify dtype=float

for i, emb in enumerate(df1['embedding']):
    # Check if emb is a list/array and not None before processing
    if emb is None or not isinstance(emb, list):
        continue
    # Convert the list to a 2D NumPy array for cosine_similarity
    emb_np = np.array(emb, dtype=float).reshape(1, -1) # Specify dtype=float

    # Check if eligible_embeddings_clean is valid for comparison
    if eligible_embeddings_clean.size == 0:
        continue # Skip if there are no embeddings to compare against

    similarities = cosine_similarity(emb_np, eligible_embeddings_clean)
    max_sim_idx = similarities.argmax()
    max_sim_val = similarities[0, max_sim_idx]

    if max_sim_val >= 0.8:
        matches.append(i)
        matched_names.append(eligible_players_clean[max_sim_idx])

filtered_df = df1.iloc[matches].copy()
filtered_df['Matched Name'] = matched_names
result_df = pd.merge(
            filtered_df,
            df2[[player_col, minutes_col]],
            left_on='Matched Name',
            right_on=player_col,
            how='left'
    )


cols_to_keep = ["Matched Name", "Age", "Team", minutes_col, "Estimated Value"]
result_df = result_df[cols_to_keep]

result_df = result_df.drop_duplicates(subset=['Matched Name'], keep='first')
result_df.to_csv("data/filtered_players2.csv", index=False)