import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

def run_migration():
    # Get Railway database URL
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    if not DATABASE_URL:
        print("❌ DATABASE_URL not found")
        return
    
    engine = create_engine(DATABASE_URL)
    
    # SQL to add missing columns
    migration_sql = [
        "ALTER TABLE tax_returns ADD COLUMN IF NOT EXISTS tax_owed FLOAT DEFAULT 0;",
        "ALTER TABLE tax_returns ADD COLUMN IF NOT EXISTS refund_amount FLOAT DEFAULT 0;",
        "ALTER TABLE tax_returns ADD COLUMN IF NOT EXISTS amount_owed FLOAT DEFAULT 0;",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS processing_status VARCHAR DEFAULT 'pending';",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_type VARCHAR;",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS extracted_data JSONB;",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS processing_error TEXT;",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS processed_at TIMESTAMP;",
        
        """
        CREATE TABLE IF NOT EXISTS document_suggestions (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES documents(id),
            field_name VARCHAR,
            suggested_value FLOAT,
            description VARCHAR,
            confidence FLOAT,
            is_accepted BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    ]
    
    try:
        with engine.connect() as conn:
            # Use transaction
            with conn.begin():
                for sql in migration_sql:
                    try:
                        print(f"Executing: {sql[:50]}...")
                        conn.execute(text(sql))
                    except Exception as e:
                        print(f"Note: {e}")
                        
            print("✅ Migration completed!")
            
    except SQLAlchemyError as e:
        print(f"❌ Migration failed: {e}")

if __name__ == "__main__":
    print("🚀 Running production migration...")
    run_migration()
