import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

class QuranFruitAnalyzer:
    def __init__(self, csv_path):
        try:
            # Read data with appropriate encoding
            self.df = pd.read_csv(csv_path, encoding='utf-8')
            # Drop any rows with NaN values
            self.df = self.df.dropna()
            # Normalize Arabic text
            self.df['verse_text'] = self.df['verse_text'].apply(lambda x: self._normalize_arabic(x))
            # Convert verse numbers to integers
            self.df['verse_number'] = pd.to_numeric(self.df['verse_number'], errors='coerce')
            self.vectorizer = TfidfVectorizer()
        except Exception as e:
            print(f"Error reading file: {e}")
            raise
        
    def _normalize_arabic(self, text):
        """Normalize Arabic text by removing diacritics and normalizing characters"""
        if not isinstance(text, str):
            return ''
        # Remove diacritics
        text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
        # Normalize alef
        text = re.sub('[إأآا]', 'ا', text)
        # Normalize yeh
        text = re.sub('[يى]', 'ي', text)
        # Normalize teh marbuta
        text = re.sub('ة', 'ه', text)
        return text
        
    def preprocess_data(self):
        try:
            # Convert verse text to numeric vector
            texts = self.df['verse_text'].astype(str)
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        except Exception as e:
            print(f"Error processing data: {e}")
            raise
        
    def find_related_verses(self, fruit_name, top_n=5):
        try:
            query_vec = self.vectorizer.transform([fruit_name])
            similarity = cosine_similarity(query_vec, self.tfidf_matrix)
            # Only get indices where we have valid data
            valid_indices = similarity[0].argsort()[-top_n:][::-1]
            results = self.df.iloc[valid_indices]
            # Return only non-empty results
            return results[results['verse_text'].notna() & results['surah'].notna()]
        except Exception as e:
            print(f"Error searching verses: {e}")
            return pd.DataFrame()
    
    def analyze_fruits_frequency(self):
        fruit_counts = {}
        try:
            # Dictionary of fruits with their variations including Unicode characters
            fruit_variations = {
                'Palm/نخل': ['نخل', 'نخيل', 'النخل', 'النخيل', 'نخلة', 'ﻧﺨﻞ', 'ﻧﺨﯿﻞ'],
                'Grape/عنب': ['عنب', 'اعناب', 'الأعناب', 'العنب', 'أعناب', 'عنابا'],
                'Pomegranate/رمان': ['رمان', 'الرمان', 'رمّان', 'الرمّان', 'رمّانٌ', 'والرمّان'],
                'Olive/زيتون': ['زيتون', 'الزيتون', 'زيتونة', 'زيتوناً', 'والزيتون'],
                'Fig/تين': ['تين', 'التين', 'والتين']
            }
            
            for fruit_name, variations in fruit_variations.items():
                count = 0
                for variant in variations:
                    # Using case-insensitive search and handling Arabic characters
                    mask = self.df['verse_text'].str.contains(
                        variant, 
                        case=False, 
                        na=False, 
                        regex=True
                    )
                    count += mask.sum()
                fruit_counts[fruit_name] = count
            
            return fruit_counts
        except Exception as e:
            print(f"Error in frequency analysis: {e}")
            print(f"Current text being processed: {self.df['verse_text'].iloc[0]}")  # Debug line
            return {}

def main():
    try:
        analyzer = QuranFruitAnalyzer('quran_verses.csv')
        analyzer.preprocess_data()
        
        while True:
            print("\n=== Quranic Fruits Smart Analysis System ===")
            print("1. Search verses related to fruit")
            print("2. Show fruit frequency analysis")
            print("3. Exit")
            
            choice = input("\nPlease select an option: ")
            
            if choice == '1':
                fruit = input("Enter fruit name: ")
                related_verses = analyzer.find_related_verses(fruit)
                if not related_verses.empty:
                    print("\nRelated verses:")
                    for _, row in related_verses.iterrows():
                        print(f"Surah {row['surah']}, Verse {row['verse_number']}: {row['verse_text']}")
                else:
                    print("No verses found.")
                    
            elif choice == '2':
                frequencies = analyzer.analyze_fruits_frequency()
                print("\nFruit frequencies in Quran:")
                for fruit, count in frequencies.items():
                    print(f"{fruit}: {count} times")
                    
            elif choice == '3':
                print("Program ended.")
                break
                
            else:
                print("Invalid option!")

    except Exception as e:
        print(f"General program error: {e}")

if __name__ == "__main__":
    main()
