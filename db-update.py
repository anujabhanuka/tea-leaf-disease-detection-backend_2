import sqlite3

# Connect to database file
conn = sqlite3.connect("app_v3.db")
cursor = conn.cursor()

# Enable WAL mode
cursor.execute("PRAGMA journal_mode=WAL;")
cursor.execute("PRAGMA synchronous=NORMAL;")

print("WAL mode enabled:", cursor.fetchone())
conn.close()