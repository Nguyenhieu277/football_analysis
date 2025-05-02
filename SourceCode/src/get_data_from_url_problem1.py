import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import numpy as np
import logging
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class FBrefScraper:
    def __init__(self):
        self.base_url = "https://fbref.com"
        self.premier_league_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
        self.all_player_data = []
        self.setup_selenium()
        
    def setup_selenium(self):
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
        try:
            self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            self.driver.set_window_size(1920, 1080)
            self.driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        except Exception as e:
            logger.error(f"Failed to initialize Selenium WebDriver: {e}")
            raise
        
    def get_soup(self, url, retries=3):
        for i in range(retries):
            try:
                self.driver.get(url)
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "table"))
                    )
                except:
                    logger.warning("Timeout waiting for table to load, proceeding anyway")
                page_source = self.driver.page_source
                return BeautifulSoup(page_source, 'html.parser')
            except Exception as e:
                logger.warning(f"Selenium attempt {i+1}/{retries} failed: {e}")
        logger.error(f"Failed to fetch {url} after {retries} attempts")
        return None

    def get_team_links(self):
        soup = self.get_soup(self.premier_league_url)
        if not soup:
            return []
        team_links = []
        for link in soup.select('table.stats_table a'):
            if '/squads/' in link.get('href'):
                team_url = urljoin(self.base_url, link.get('href'))
                if team_url not in team_links:
                    team_links.append(team_url)
        return team_links
    
    def get_stat_value(self, row, stat, numeric=False):
        cell = row.select_one(f'td[data-stat="{stat}"], th[data-stat="{stat}"]')
        if not cell:
            return "N/a"
        value = cell.text.strip()
        if value == '' or value == '-':
            return "N/a"
        if numeric:
            try:
                value = value.replace(',', '')
                return float(value)
            except ValueError:
                return "N/a"
        return value
    
    def extract_standard_stats(self, row, player_data):
        player_data['nation'] = self.get_stat_value(row, 'nationality')
        player_data['position'] = self.get_stat_value(row, 'position')
        player_data['age'] = self.get_stat_value(row, 'age')
        raw_age = self.get_stat_value(row, 'age')
        try:
            player_data['age'] = int(raw_age.split('-')[0]) if raw_age != "N/a" else "N/a"
        except (ValueError, AttributeError, IndexError):
            player_data['age'] = "N/a"
        player_data['matches_played'] = self.get_stat_value(row, 'games', True)
        player_data['starts'] = self.get_stat_value(row, 'games_starts', True)
        player_data['minutes'] = self.get_stat_value(row, 'minutes', True)
        player_data['goals'] = self.get_stat_value(row, 'goals', True)
        player_data['assists'] = self.get_stat_value(row, 'assists', True)
        player_data['yellow_cards'] = self.get_stat_value(row, 'cards_yellow', True)
        player_data['red_cards'] = self.get_stat_value(row, 'cards_red', True)
        player_data['xG'] = self.get_stat_value(row, 'xg', True)
        player_data['xAG'] = self.get_stat_value(row, 'xg_assist', True)
        player_data['PrgC'] = self.get_stat_value(row, 'progressive_carries', True)
        player_data['PrgP'] = self.get_stat_value(row, 'progressive_passes', True)
        player_data['Prgr'] = self.get_stat_value(row, 'progressive_passes_received', True)
    
    def extract_shooting_stats(self, row, player_data):
        player_data['SoT%'] = self.get_stat_value(row, 'shots_on_target_pct', True)
        player_data['SoT/90'] = self.get_stat_value(row, 'shots_on_target_per90', True)
        player_data['G/sh'] = self.get_stat_value(row, 'goals_per_shot', True)
        player_data['Dist'] = self.get_stat_value(row, 'average_shot_distance', True)
    
    def extract_passing_stats(self, row, player_data):
        player_data['Cmp'] = self.get_stat_value(row, 'passes_completed', True)
        player_data['Cmp%'] = self.get_stat_value(row, 'passes_pct', True)
        player_data['TotDist'] = self.get_stat_value(row, 'passes_total_distance', True)
        player_data['Short_Cmp%'] = self.get_stat_value(row, 'passes_pct_short', True)
        player_data['Medium_Cmp%'] = self.get_stat_value(row, 'passes_pct_medium', True)
        player_data['Long_Cmp%'] = self.get_stat_value(row, 'passes_pct_long', True)
        player_data['KP'] = self.get_stat_value(row, 'assisted_shots', True)
        player_data['1/3'] = self.get_stat_value(row, 'passes_into_final_third', True)
        player_data['PPA'] = self.get_stat_value(row, 'passes_into_penalty_area', True)
        player_data['CrsPA'] = self.get_stat_value(row, 'crosses_into_penalty_area', True)
        player_data['PrgP/passing'] = self.get_stat_value(row, 'progressive_passes', True)
    
    def extract_gca_stats(self, row, player_data):
        player_data['SCA'] = self.get_stat_value(row, 'sca', True)
        player_data['SCA90'] = self.get_stat_value(row, 'sca_per90', True)
        player_data['GCA'] = self.get_stat_value(row, 'gca', True)
        player_data['GCA90'] = self.get_stat_value(row, 'gca_per90', True)
    
    def extract_defense_stats(self, row, player_data):
        player_data['Tkl'] = self.get_stat_value(row, 'tackles', True)
        player_data['TklW'] = self.get_stat_value(row, 'tackles_won', True)
        player_data['Att'] = self.get_stat_value(row, 'challenges', True)
        player_data['Lost'] = self.get_stat_value(row, 'challenges_lost', True)
        player_data['Blocks'] = self.get_stat_value(row, 'blocks', True)
        player_data['Sh'] = self.get_stat_value(row, 'blocked_shots', True)
        player_data['Pass'] = self.get_stat_value(row, 'blocked_passes', True)
        player_data['Int'] = self.get_stat_value(row, 'interceptions', True)
    
    def extract_possession_stats(self, row, player_data):
        player_data['Touches'] = self.get_stat_value(row, 'touches', True)
        player_data['Def Pen'] = self.get_stat_value(row, 'touches_def_pen_area', True)
        player_data['Def 3rd'] = self.get_stat_value(row, 'touches_def_3rd', True)
        player_data['Mid 3rd'] = self.get_stat_value(row, 'touches_mid_3rd', True)
        player_data['Att 3rd'] = self.get_stat_value(row, 'touches_att_3rd', True)
        player_data['Att Pen'] = self.get_stat_value(row, 'touches_att_pen_area', True)
        player_data['Att'] = self.get_stat_value(row, 'take_ons_attempted', True)
        player_data['Succ%'] = self.get_stat_value(row, 'take_ons_won_pct', True)
        player_data['Tkld%'] = self.get_stat_value(row, 'take_ons_tackled_pct', True)
        player_data['Carries'] = self.get_stat_value(row, 'carries', True)
        player_data['ProDist'] = self.get_stat_value(row, 'carries_distance', True)
        player_data['ProgC'] = self.get_stat_value(row, 'progressive_carries', True)
        player_data['1/3'] = self.get_stat_value(row, 'carries_into_final_third', True)
        player_data['CPA'] = self.get_stat_value(row, 'carries_into_penalty_area', True)
        player_data['Mis'] = self.get_stat_value(row, 'miscontrols', True)
        player_data['Dis'] = self.get_stat_value(row, 'dispossessed', True)
        player_data['Rec'] = self.get_stat_value(row, 'passes_received', True)
        player_data['PrgR'] = self.get_stat_value(row, 'progressive_passes_received', True)
    
    def extract_misc_stats(self, row, player_data):
        player_data['Fls'] = self.get_stat_value(row, 'fouls', True)
        player_data['Fld'] = self.get_stat_value(row, 'fouled', True)
        player_data['Off'] = self.get_stat_value(row, 'offsides', True)
        player_data['Crs'] = self.get_stat_value(row, 'crosses', True)
        player_data['Recov'] = self.get_stat_value(row, 'ball_recoveries', True)
        player_data['Won'] = self.get_stat_value(row, 'aerials_won', True)
        player_data['Lost'] = self.get_stat_value(row, 'aerials_lost', True)
        player_data['Won%'] = self.get_stat_value(row, 'aerials_won_pct', True)
    
    def extract_keeper_stats(self, row, player_data):
        player_data['GA90'] = self.get_stat_value(row, 'gk_goals_against_per90', True)
        player_data['Save%'] = self.get_stat_value(row, 'gk_save_pct', True)
        player_data['CS%'] = self.get_stat_value(row, 'gk_clean_sheets_pct', True)
        player_data['PSave%'] = self.get_stat_value(row, 'gk_pens_save_pct', True)
    
    def find_stat_tables(self, soup):
        tables = {}
        table_ids = {
            'standard': ['stats_standard_9'],
            'shooting': ['stats_shooting_9'],
            'passing': ['stats_passing_9', 'stats_passing_types_9'],
            'possession': ['stats_possession_9'],
            'defense': ['stats_defense_9'],
            'gca': ['stats_gca_9'],
            'misc': ['stats_misc_9'],
            'keeper': ['stats_keeper_9', 'stats_keeper_adv_9']
        }
        all_tables = soup.find_all('table')
        table_ids_found = [table.get('id', 'no-id') for table in all_tables]
        for stat_type, ids in table_ids.items():
            for id in ids:
                table = soup.select_one(f'table#{id}')
                if table:
                    tables[stat_type] = table
                    break
        return tables
    
    def get_stat_page_links(self, soup):
        links = {}
        nav_links = soup.select('div#inner_nav a')
        for link in nav_links:
            href = link.get('href')
            text = link.text.lower()
            if not href:
                continue
            if 'shooting' in href or 'shooting' in text:
                links['shooting'] = urljoin(self.base_url, href)
            elif 'passing' in href or 'passing' in text:
                links['passing'] = urljoin(self.base_url, href)
            elif 'possession' in href or 'possession' in text:
                links['possession'] = urljoin(self.base_url, href)
            elif 'defense' in href or 'defensive' in text:
                links['defense'] = urljoin(self.base_url, href)
            elif 'gca' in href or 'creation' in text:
                links['gca'] = urljoin(self.base_url, href)
            elif 'misc' in href or 'miscellaneous' in text:
                links['misc'] = urljoin(self.base_url, href)
            elif 'keeper' in href or 'goalkeeping' in text:
                links['keeper'] = urljoin(self.base_url, href)
        return links
    
    def extract_player_data_from_table(self, table, stat_type, team_name):
        players = []
        all_rows = table.select('tbody tr')
        for row in all_rows:
            if 'class' in row.attrs and any(c in row['class'] for c in ['thead', 'spacer']):
                continue
            player_cell = row.select_one('th[data-stat="player"] a')
            if not player_cell:
                player_cell = row.select_one('th a')
            if not player_cell:
                continue
            player_name = player_cell.text.strip()
            player_data = {'player': player_name, 'team': team_name}
            if stat_type == 'standard':
                self.extract_standard_stats(row, player_data)
            elif stat_type == 'shooting':
                self.extract_shooting_stats(row, player_data)
            elif stat_type == 'passing':
                self.extract_passing_stats(row, player_data)
            elif stat_type == 'possession':
                self.extract_possession_stats(row, player_data)
            elif stat_type == 'defense':
                self.extract_defense_stats(row, player_data)
            elif stat_type == 'gca':
                self.extract_gca_stats(row, player_data)
            elif stat_type == 'misc':
                self.extract_misc_stats(row, player_data)
            elif stat_type == 'keeper':
                self.extract_keeper_stats(row, player_data)
            players.append(player_data)
        return players

    def scrape_team(self, team_url):
        team_soup = self.get_soup(team_url)
        if not team_soup:
            return []
        team_elem = team_soup.select_one('h1[itemprop="name"] span')
        if team_elem:
            team_name = team_elem.text.strip()
        else:
            title_elem = team_soup.select_one('h1')
            team_name = title_elem.text.strip() if title_elem else "Unknown"
        player_data_by_name = {}
        tables = self.find_stat_tables(team_soup)
        if not tables:
            tables_alt = {}
            for table in team_soup.find_all('table'):
                if 'stats_standard' in table.get('id', ''):
                    tables_alt['standard'] = table
            if tables_alt:
                tables = tables_alt
        if not tables:
            return []
        for stat_type, table in tables.items():
            players = self.extract_player_data_from_table(table, stat_type, team_name)
            for player in players:
                name = player['player']
                if name not in player_data_by_name:
                    player_data_by_name[name] = player
                else:
                    player_data_by_name[name].update(player)
        stat_links = self.get_stat_page_links(team_soup)
        for stat_type, link in stat_links.items():
            if stat_type in tables:
                continue
            stat_soup = self.get_soup(link)
            if not stat_soup:
                    continue
            stat_tables = self.find_stat_tables(stat_soup)
            for table_type, table in stat_tables.items():
                players = self.extract_player_data_from_table(table, table_type, team_name)
                for player in players:
                    name = player['player']
                    if name not in player_data_by_name:
                        player_data_by_name[name] = player
                    else:
                        player_data_by_name[name].update(player)
            time.sleep(random.uniform(1, 2))
        result = list(player_data_by_name.values())
        return result

    def run(self):
        try:
            team_links = self.get_team_links()
            if not team_links:
                logger.error("No team links found. Aborting.")
                return None
            all_players = []
            for i, team_url in enumerate(team_links):
                team_players = self.scrape_team(team_url)
                if team_players:
                    all_players.extend(team_players)
                time.sleep(random.uniform(2, 4))
            if not all_players:
                return None
            df = pd.DataFrame(all_players)
            if 'minutes' in df.columns:
                df['minutes'] = df['minutes'].astype(str).str.replace(',', '')
                df['minutes'] = pd.to_numeric(df['minutes'], errors='coerce')
                df = df.dropna(subset=['minutes']).copy()
                before_len = len(df)
                df = df[df['minutes'] > 90].copy()
            else:
                logger.warning("'minutes' column not found in data. Skipping filter.")
            if 'player' in df.columns:
                df = df.sort_values('player')
            df = df.replace({np.nan: "N/a"})
            df.to_csv('data/results.csv', index=False)
            return df
        finally:
            if hasattr(self, 'driver'):
                logger.info("Closing Selenium WebDriver")
            self.driver.quit()

if __name__ == "__main__":
    scraper = FBrefScraper()
    scraper.run()
