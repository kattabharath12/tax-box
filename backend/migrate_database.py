import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

def migrate_database():
    """Add new columns to existing tables"""
    
    migrations = [
        # Users table updates
        "ALTER TABLE users ADD COLUMN filing_status VARCHAR DEFAULT 'single'",
        "ALTER TABLE users ADD COLUMN state VARCHAR",
        
        # Tax returns table updates
        "ALTER TABLE tax_returns ADD COLUMN state_income FLOAT DEFAULT 0",
        "ALTER TABLE tax_returns ADD COLUMN state_deductions FLOAT DEFAULT 0", 
        "ALTER TABLE tax_returns ADD COLUMN state_tax_owed FLOAT DEFAULT 0",
        "ALTER TABLE tax_returns ADD COLUMN state_withholdings FLOAT DEFAULT 0",
        "ALTER TABLE tax_returns ADD COLUMN state_refund FLOAT DEFAULT 0",
        "ALTER TABLE tax_returns ADD COLUMN state_amount_owed FLOAT DEFAULT 0"
    ]
    
    with engine.connect() as conn:
        for migration in migrations:
            try:
                conn.execute(text(migration))
                conn.commit()
                print(f"✓ Successfully executed: {migration}")
            except Exception as e:
                if "already exists" in str(e) or "duplicate column" in str(e).lower():
                    print(f"⚠ Column already exists: {migration}")
                else:
                    print(f"✗ Error: {migration} - {e}")
    
    print("\n✅ Database migration completed!")

if __name__ == "__main__":
    migrate_database()
