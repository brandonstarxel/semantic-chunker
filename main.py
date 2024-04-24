# print("Sample code")

import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('example.db')

# Create a cursor object
cursor = conn.cursor()

# Create table
cursor.execute('''CREATE TABLE IF NOT EXISTS example_table
             (date text, trans text, symbol text, qty real, price real)''')

# Insert a row of data
cursor.execute("INSERT INTO example_table VALUES ('2006-01-05','BUY','RHAT',100,35.14, peee pee ppoo)")

# Save (commit) the changes
conn.commit()

# Close the connection
conn.close()
