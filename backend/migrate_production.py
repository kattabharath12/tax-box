# migrate_production.py
import os
from sqlalchemy import create_engine, text

def run_migration():
    """Run database migration to add filing status columns"""
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    if not DATABASE_URL:
        print("❌ No DATABASE_URL found")
        return
    
    try:
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as conn:
            # Start a transaction
            trans = conn.begin()
            try:
                # Add missing columns
                commands = [
                    "ALTER TABLE tax_returns ADD COLUMN IF NOT EXISTS filing_status VARCHAR DEFAULT 'single';",
                    "ALTER TABLE tax_returns ADD COLUMN IF NOT EXISTS spouse_name VARCHAR;",
                    "ALTER TABLE tax_returns ADD COLUMN IF NOT EXISTS spouse_ssn VARCHAR;",
                    "ALTER TABLE tax_returns ADD COLUMN IF NOT EXISTS spouse_has_income BOOLEAN DEFAULT FALSE;",
                    "ALTER TABLE tax_returns ADD COLUMN IF NOT EXISTS spouse_itemizes BOOLEAN DEFAULT FALSE;",
                    "ALTER TABLE tax_returns ADD COLUMN IF NOT EXISTS qualifying_person_name VARCHAR;",
                    "ALTER TABLE tax_returns ADD COLUMN IF NOT EXISTS qualifying_person_relationship VARCHAR;",
                    "ALTER TABLE tax_returns ADD COLUMN IF NOT EXISTS lived_with_taxpayer BOOLEAN DEFAULT FALSE;"
                ]
                
                for cmd in commands:
                    conn.execute(text(cmd))
                
                # Commit the transaction
                trans.commit()
                print("✅ Migration completed successfully")
                
            except Exception as e:
                trans.rollback()
                raise e
                
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    run_migration()
