import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sqlite3
import pandas as pd
from src.sql_queries import build_database, get_completion_rate_by_genre, execute_query
from src.data_generator import save_generated_data
from src.data_processing import process_and_clean_data

class TestSQLQueries(unittest.TestCase):
    def test_sql_database_build(self):
        raw_dir = "data/raw"
        proc_dir = "data/processed"
        db_path = "database/reading_data.db"
        schema_path = "database/schema.sql"
        
        save_generated_data(raw_dir)
        process_and_clean_data(raw_dir, proc_dir)
        build_database(proc_dir, db_path, schema_path)
        
        self.assertTrue(os.path.exists(db_path))
        df_res = get_completion_rate_by_genre(db_path)
        self.assertIsInstance(df_res, pd.DataFrame)
        self.assertIn('genre', df_res.columns)

if __name__ == "__main__":
    unittest.main()
