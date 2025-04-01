# init_db.py
from dotenv import load_dotenv
load_dotenv()

import psycopg2
import os
from urllib.parse import urlparse

# Parse DATABASE_URL
db_url = os.getenv("DATABASE_URL")
if not db_url:
    raise Exception("❌ DATABASE_URL not set")

parsed = urlparse(db_url)
DB_NAME = parsed.path[1:]
DB_USER = parsed.username
DB_PASSWORD = parsed.password
DB_HOST = parsed.hostname
DB_PORT = parsed.port


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
