# init_db.py

import psycopg2
import os

DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT", 5432)

def create_predictions_table():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP,
                predicted INTEGER NOT NULL,
                true_label INTEGER NOT NULL,
                confidence FLOAT NOT NULL
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Table 'predictions' created or already exists.")
    except Exception as e:
        print(f"❌ Error creating table: {e}")

if __name__ == "__main__":
    create_predictions_table()
