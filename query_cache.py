import sqlite3

conn = sqlite3.connect(".ragmark_cache/judge_cache.db")
cursor = conn.execute("SELECT * FROM judge_cache;")
rows = cursor.fetchall()
for row in rows:
    print(row)
conn.close()
