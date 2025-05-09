import pandas as pd
import requests
import time
import random
import google.generativeai as genai
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import ast


load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
def get_embedding(text, model='models/gemini-embedding-exp-03-07'):
        
    result = genai.embed_content(model=model, content=text, task_type="RETRIEVAL_DOCUMENT")
    time.sleep(1.5)
    return result["embedding"] 
    
class TransferScraper:
    def __init__(self):
        self.api_url = 'https://www.footballtransfers.com/us/values/actions/most-valuable-football-players/overview'
        self.lastpage = 30
        self.all_transfers = []
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Referer": "https://www.footballtransfers.com/us/values/players/most-valuable-soccer-players/playing-in-uk-premier-league",
            "X-Requested-With": "XMLHttpRequest",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
        }
        
        self.payload = {
            "orderBy": "estimated_value",
            "orderByDescending": 1,
            "page": 1,
            "pages": 0,
            "pageItems": 25,
            "positionGroupId": "all",
            "mainPositionId": "all",
            "playerRoleId": "all",
            "age": "all",
            "countryId": "all",
            "tournamentId": 31  
        }
            
    def fetch_page_data(self, page_num):
        try:
            self.payload["page"] = page_num
            wait = random.uniform(0.5, 3.5)
            time.sleep(wait)
            
            resp = requests.post(self.api_url, headers=self.headers, data=self.payload)
            if resp.status_code == 200:
                json_data = resp.json()
                if "records" in json_data:
                    player_records = json_data["records"]
                    data_frame = pd.DataFrame(player_records)
                    cols = ["player_name", "age", "team_name", "estimated_value"]
                    data_frame = data_frame[cols]
                    self.all_transfers.append(data_frame)
                    print(f"Successfully added {len(player_records)} players from page {page_num}")
                    return True
            return False
        except Exception as e:
            return False

    def parse_embedding(self, embedding_str):
        
        if pd.isna(embedding_str):
            return None
        try:
            return ast.literal_eval(embedding_str)
        except (ValueError, SyntaxError):
            return None
    
    def filter_players(self, similarity_threshold=0.8, min_minutes=900):
        if not self.all_transfers:
            return None

        
        df = pd.concat(self.all_transfers)
        df.columns = ["Player", "Age", "Team", "Estimated Value"]
        df = df.dropna()

        
        print("Generating embeddings for scraped players")
        embeddings = []
        for idx, player_name in enumerate(df['Player']):
            print(f"Embedding {idx+1}/{len(df)}: {player_name}")
            emb = get_embedding(player_name)
            embeddings.append(emb)

        df['embedding'] = embeddings
        df.to_csv(r'D:\work\football_analysis\embed_df.csv')

        
        player_df = pd.read_csv(r"D:\work\football_analysis\SourceCode\data\results.csv")

        
        player_df_filtered = player_df[player_df['minutes'] > min_minutes]

        
        print("Generating embeddings for eligible players")
        eligible_embeddings = []
        for idx, player_name in enumerate(player_df_filtered['player']):
            print(f"Embedding {idx+1}/{len(player_df_filtered)}: {player_name}")
            emb = get_embedding(player_name)
            eligible_embeddings.append(emb)

        player_df_filtered['embedding'] = eligible_embeddings
        player_df_filtered.to_csv(r'D:\work\football_analysis\embed_df2.csv')

        
        eligible_embeddings_clean = []
        eligible_players_clean = []
        
        for idx, emb in enumerate(eligible_embeddings):
            if emb is not None:
                eligible_embeddings_clean.append(emb)
                eligible_players_clean.append(player_df_filtered.iloc[idx]['player'])

        if not eligible_embeddings_clean:         
            return None

        eligible_embeddings_clean = np.array(eligible_embeddings_clean)

       
        matches = []
        matched_names = []

        for i, emb in enumerate(df['embedding']):
            if emb is None:
                continue
                
            emb = np.array(emb).reshape(1, -1)
            
            similarities = cosine_similarity(emb, eligible_embeddings_clean)
            max_sim_idx = similarities.argmax()
            max_sim_val = similarities[0, max_sim_idx]
            
            if max_sim_val >= similarity_threshold:
                matches.append(i)
                matched_names.append(eligible_players_clean[max_sim_idx])

        
        filtered_df = df.iloc[matches].copy()
        filtered_df['Matched Name'] = matched_names
        
        result_df = pd.merge(
            filtered_df,
            player_df_filtered[['player', 'minutes']],
            left_on='Matched Name',
            right_on='player',
            how='left'
        )

        
        cols_to_keep = ["Matched Name", "Age", "Team", "minutes", "Estimated Value"]
        result_df = result_df[cols_to_keep]
        result_df = result_df.drop_duplicates(subset=['Matched Name'], keep='first')

    
        result_df.to_csv(r"D:\work\football_analysis\SourceCode\data\filtered_players.csv", index=False)
        return result_df

    def run(self):
        try:
            for page in range(1, self.lastpage + 1):
                success = self.fetch_page_data(page)
                if not success:
                    pass
            
            df = self.filter_players()
            if df is not None:
                print(df)
        except Exception as e:
            print(f"Error in run method: {e}")
            pass

if __name__ == "__main__":
    scraper = TransferScraper()
    scraper.run()
