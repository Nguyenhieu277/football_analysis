import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import re


class DataProcessor:
    def __init__(self, input_file_path):
        
        self.input_file_path = input_file_path
        self.df = None
        self.filtered_df = None
        self.df_final = None
        self.le = LabelEncoder()
        self.mlb = MultiLabelBinarizer()
        self.position_classes = None
        
    def load_data(self):

        self.df = pd.read_csv(self.input_file_path)
        return self
        
    def clean_data(self):
    
        self.filtered_df = self.df.replace('N/a', 0)
        self.filtered_df['Value'] = self.filtered_df['Value'].replace({'€': '', 'M': ''}, regex=True).astype(float)
        return self
    
    def split_nationality(self, nation_str):
        
        if not nation_str:
            return []
        
        nation = nation_str.split()
        return str(nation[1])
    
    def process_nationality(self):
        
        self.filtered_df['nation'] = self.filtered_df['nation'].apply(self.split_nationality)
        self.filtered_df['nation_label_encoded'] = self.le.fit_transform(self.filtered_df['nation'])
        return self
    
    def split_positions(self, position_str):
        
        if not position_str:
            return []
        
        positions = re.split(r'[,]\s*|\s*[,]', position_str.strip())
        return [pos.strip() for pos in positions if pos.strip()]
    
    def process_positions(self):
        
        self.filtered_df['position_list'] = self.filtered_df['position'].apply(self.split_positions)
        
        
        position_encoded_array = self.mlb.fit_transform(self.filtered_df['position_list'])
        self.position_classes = self.mlb.classes_
        
        position_encoded_df = pd.DataFrame(
            position_encoded_array, 
            columns=[f'pos_{cls}' for cls in self.position_classes], 
            index=self.df.index
        )
        
        
        self.df_final = pd.concat([
            self.filtered_df.drop(['position', 'position_list'], axis=1), 
            position_encoded_df
        ], axis=1)
        
        return self
    
    
    def save_data(self, output_file_path):
       
        self.df_final.to_csv(output_file_path, index=False)
        return self
    
    def print_summary(self):
        print(self.df_final)
        print("\n--- Các lớp (quốc gia) đã được mã hóa ---")
        print(list(self.le.classes_))
        return self
    
    def process_pipeline(self, output_file_path):
        
        return (self.load_data()
                .clean_data()
                .process_nationality()
                .process_positions()
                .save_data(output_file_path)
                .print_summary())


if __name__ == "__main__":
    processor = DataProcessor(r'data/merge_df.csv')
    processor.process_pipeline(r'D:\work\BTL\data\players_played_more_than_900m.csv')