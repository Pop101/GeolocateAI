import sqlite3
import os

# Path to the SQLite database file
db_file = "./2015_placing/placemeta_train_photo.sql"

# Check if the file exists
if not os.path.exists(db_file):
    print(f"File not found: {db_file}")
else:
    try:
        # Connect directly to the database
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if tables:
            print("Tables found in the database:", [table[0] for table in tables])
            
            # For each table, print schema and sample data
            for table_name in [table[0] for table in tables]:
                print(f"\n--- Table: {table_name} ---")
                
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                print("Schema:")
                for col in columns:
                    print(f"  {col[1]} ({col[2]})")
                
                #  32362630 rows
                
                # Print sample data (first 5 rows)
                print("Sample data:")
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
                rows = cursor.fetchall()
                for row in rows:
                    print(f"  {row}")
                    
                # All feature types
                print("All feature types:")
                cursor.execute(f"SELECT DISTINCT type FROM {table_name} ")
                rows = cursor.fetchall()
                for row in rows:
                    print(f"  {row}")
        else:
            print("No tables found in the database.")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")