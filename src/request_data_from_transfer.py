import pandas as pd
import requests
import time
import random

class TransferScraper:
    def __init__(self):
        self.api_url = 'https://www.footballtransfers.com/us/values/actions/most-valuable-football-players/overview'
        self.lastpage = 23
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
            wait = random.uniform(0.5, 1.5)
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
        
    def filter_players(self):
        if not self.all_transfers:
            return None
            
        df = pd.concat(self.all_transfers)
        df.columns = ["Player", "Age", "Team", "Estimated Value"]
        
        try:
            player_df = pd.read_csv("data/results.csv")
        except Exception as e:
            return None
            
        minutes_col = 'minutes' if 'minutes' in player_df.columns else 'Standard_Min'
        
        if minutes_col not in player_df.columns:
            return None
            
        
        player_df_filtered = player_df[player_df[minutes_col] > 900]
        
        player_col = 'player' if 'player' in player_df.columns else 'Player'
        if player_col not in player_df.columns:
            return None 
            
        eligible_players = player_df_filtered[player_col].tolist()
        
        filtered_df = df[df['Player'].isin(eligible_players)]
        
        result_df = pd.merge(
            filtered_df,
            player_df_filtered[[player_col, minutes_col]], 
            left_on='Player', 
            right_on=player_col,
            how='left'
        )
        
        cols_to_keep = ["Player", "Age", "Team", minutes_col, "Estimated Value"] 
        result_df = result_df[cols_to_keep]
        
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
                df.to_csv("data/filtered_players.csv", index=False)
        except Exception as e:
            pass

if __name__ == "__main__":
    scraper = TransferScraper()
    scraper.run()
