import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

host = os.getenv("MYSQL_HOST")
port = os.getenv("MYSQL_PORT", "3306")
user = os.getenv("MYSQL_USER")
password = os.getenv("MYSQL_PASSWORD")
database = os.getenv("MYSQL_DATABASE")

print("Configuration loaded.")
print("Host:", host)
print("Port:", port)
print("User:", user)
print("Database:", database)

DATABASE_URL = (
    f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
)

try:
    print("Creating MySQL engine...")

    engine = create_engine(
        DATABASE_URL,
        connect_args={"connect_timeout": 5}
    )

    print("Trying to connect to MySQL...")

    with engine.connect() as connection:

        result = connection.execute(
            text("SELECT DATABASE();")
        )

        database_name = result.fetchone()[0]

        print("MySQL connected successfully!")
        print("Current database:", database_name)

except Exception as e:
    print("MySQL connection failed!")
    print("Error:", repr(e))