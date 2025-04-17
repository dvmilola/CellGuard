import sqlite3
import os

# Print current directory for reference
print(f"Current directory: {os.getcwd()}")
print(f"Database file exists: {os.path.exists('instance/crisis_predictions.db')}")

# Connect to the database
conn = sqlite3.connect('instance/crisis_predictions.db')
cursor = conn.cursor()

# Get list of tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("\nTables in the database:", tables)

# Show schema for each table
for table in tables:
    table_name = table[0]
    print(f"\nSchema for table {table_name}:")
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    for column in columns:
        print(f"  {column}")

# Count records in each table
for table in tables:
    table_name = table[0]
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    count = cursor.fetchone()[0]
    print(f"\nNumber of records in {table_name}: {count}")

# If there are records in the prediction table, show a sample
if tables and 'prediction' in [t[0] for t in tables]:
    cursor.execute("SELECT * FROM prediction LIMIT 5;")
    records = cursor.fetchall()
    if records:
        print("\nSample records from prediction table:")
        # Get column names
        cursor.execute("PRAGMA table_info(prediction);")
        columns = [column[1] for column in cursor.fetchall()]
        print("  Columns:", columns)
        
        # Print records
        for record in records:
            print(f"  {record}")

# Close the connection
conn.close()
