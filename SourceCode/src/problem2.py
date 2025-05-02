import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re

df = pd.read_csv(r"D:\work\football_analysis\SourceCode\data\results.csv")

percent_cols = ['Won%', 'Save%', 'CS%', 'PSave%']
for col in percent_cols:
    df[col] = (
        df[col]
        .replace('N/a', np.nan)
        .str.replace(r'[^\d.]', '', regex=True)
        .pipe(pd.to_numeric, errors='coerce')
        / 100
    )

df['GA90'] = pd.to_numeric(df['GA90'], errors='coerce')

os.makedirs('histograms', exist_ok=True)

stats_columns = df.columns[4:].tolist()
for stat in stats_columns:
    df[stat] = pd.to_numeric(df[stat], errors='coerce')

def generate_top_bottom_3():
    with open('data/top_3.txt', 'w', encoding='utf-8') as f:
        for stat in stats_columns:
            df[stat] = pd.to_numeric(df[stat], errors='coerce')

            valid_count = df[stat].count()
            
            if valid_count < 3:
                continue

            f.write(f"\n{'='*25} {stat.upper()} {'='*25}\n")
            
            top3 = df[['player', 'team', stat]].nlargest(3, stat)
            f.write("\nTop 3:\n")
            for _, row in top3.iterrows():
                f.write(f"{row['player']} ({row['team']}): {row[stat]:.2f}\n")
            
            bottom3 = df[['player', 'team', stat]].nsmallest(3, stat)
            f.write("\nBottom 3:\n")
            for _, row in bottom3.iterrows():
                f.write(f"{row['player']} ({row['team']}): {row[stat]:.2f}\n")
def calculate_statistics():
    team_cols = df['team'].unique().tolist()
    
    results = pd.DataFrame(columns=['Statistic', 'all'] + team_cols)
    
    for stat in stats_columns:
        
        all_median = df[stat].median()
        all_mean = df[stat].mean()
        all_std = df[stat].std()

        
        new_rows_data = {
            'Statistic': [f'Median of {stat}', f'Mean of {stat}', f'Std of {stat}'],
            'all': [all_median, all_mean, all_std]
        }

       
        for team in team_cols:
            team_data = df[df['team'] == team][stat]
            new_rows_data[team] = [team_data.median(), team_data.mean(), team_data.std()]

        
        new_rows = pd.DataFrame(new_rows_data)

        
        results = pd.concat([results, new_rows], ignore_index=True)

    
    transposed = results.set_index('Statistic').transpose()
    transposed.index.name = 'Team/Stat'
    transposed.reset_index(inplace=True)
    
    transposed.rename(columns={'index': 'Team/Stat'}, inplace=True)

    cols = ['Team/Stat'] + [col for col in transposed.columns if col != 'Team/Stat']
    transposed = transposed[cols]

    transposed.to_csv(r'data/results2.csv', index=False, float_format="%.2f")

def generate_histograms():
    os.makedirs('histograms', exist_ok=True)
    attack = ['xG', 'goals', 'SoT%']
    defense = ['Tkl', 'TklW', 'Def 3rd']
    for stat in attack:
        plt.figure()
        all_players_data = df[stat].dropna()
        all_players_data.hist()
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {stat} for All Players')
        
        safe_stat = re.sub(r'[^a-zA-Z0-9]', '_', stat)
        safe_filename = f'histograms/all_players_{safe_stat}.png'
        plt.savefig(safe_filename, bbox_inches='tight')
        plt.close()
    for stat in defense:
        plt.figure()
        all_players_data = df[stat].dropna()
        all_players_data.hist()
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {stat} for All Players')
        
        safe_stat = re.sub(r'[^a-zA-Z0-9]', '_', stat)
        safe_filename = f'histograms/all_players_{safe_stat}.png'
        plt.savefig(safe_filename, bbox_inches='tight')
        plt.close()

def generate_histograms2():
    os.makedirs('histograms', exist_ok=True)
    attack = ['xG', 'goals', 'SoT%']
    defense = ['Tkl', 'TklW', 'Def 3rd']
    
    for stat in attack:
        for team in df['team'].unique():
            safe_team = re.sub(r'[\n<>:"/\\|?*()]', '_', team).replace(' ', '_').strip('_')
            safe_stat = re.sub(r'[^a-zA-Z0-9]', '_', stat)
            
            plt.figure()
            team_data = df[df['team'] == team][stat]
            team_data.hist()
            plt.title(f'{team} - {stat}')
            
            safe_filename = f'histograms/{sanitize_filename(safe_team)}_{safe_stat}.png'
            plt.savefig(safe_filename, bbox_inches='tight')
            plt.close()
    for stat in defense:
        for team in df['team'].unique():
            safe_team = re.sub(r'[\n<>:"/\\|?*()]', '_', team).replace(' ', '_').strip('_')
            safe_stat = re.sub(r'[^a-zA-Z0-9]', '_', stat)
            
            plt.figure()
            team_data = df[df['team'] == team][stat]
            team_data.hist()
            plt.title(f'{team} - {stat}')
            
            safe_filename = f'histograms/{sanitize_filename(safe_team)}_{safe_stat}.png'
            plt.savefig(safe_filename, bbox_inches='tight')
            plt.close()

def analyze_data():
    team_stat = df.groupby('team')[stats_columns].mean()
    leadership = {}

    for stat in stats_columns:
        try:
            leader = team_stat[stat].idxmax()
            leadership[stat] = leader
        except:
            continue

    leader_counts = pd.Series(leadership).value_counts()
    best_team = leader_counts.idxmax()
    leader_counts.to_csv(r'data/leadership.csv')

def sanitize_filename(filename):
    invalid_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|', '\r', '\n']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    filename = filename.rstrip('. ')
    
    return filename

analyze_data()
generate_histograms()
generate_histograms2()  
calculate_statistics()
generate_top_bottom_3()