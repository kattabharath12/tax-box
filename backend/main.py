import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Auto-run migration if needed
if os.getenv("RUN_MIGRATION") == "true":
    try:
        from migrate_production import run_migration
        run_migration()
        print("✅ Migration completed on startup")
    except Exception as e:
        print(f"❌ Migration failed: {e}")

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel, EmailStr
import jwt
from passlib.context import CryptContext
import uvicorn
import shutil

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./taxbox.db")

# Add SSL configuration for PostgreSQL
connect_args = {}
if "postgresql" in DATABASE_URL:
    connect_args = {
        "sslmode": "require",
        "connect_timeout": 10,
    }
elif "sqlite" in DATABASE_URL:
    connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL, 
    connect_args=connect_args,
    pool_pre_ping=True,
    pool_recycle=300
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_cpa = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    documents = relationship("Document", back_populates="user")
    tax_returns = relationship("TaxReturn", back_populates="user")
    payments = relationship("Payment", back_populates="user")

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String)
    file_path = Column(String)
    file_type = Column(String)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    
    # Document processing fields
    processing_status = Column(String, default="pending")
    document_type = Column(String)
    extracted_data = Column(JSON)
    processing_error = Column(Text)
    processed_at = Column(DateTime)

    user = relationship("User", back_populates="documents")
    document_suggestions = relationship("DocumentSuggestion", back_populates="document")

class DocumentSuggestion(Base):
    __tablename__ = "document_suggestions"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    field_name = Column(String)
    suggested_value = Column(Float)
    description = Column(String)
    confidence = Column(Float)
    is_accepted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    document = relationship("Document", back_populates="document_suggestions")

class TaxReturn(Base):
    __tablename__ = "tax_returns"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    tax_year = Column(Integer)
    income = Column(Float)
    deductions = Column(Float)
    withholdings = Column(Float)
    tax_owed = Column(Float)
    refund_amount = Column(Float)
    amount_owed = Column(Float)
    status = Column(String, default="draft")
    created_at = Column(DateTime, default=datetime.utcnow)
    submitted_at = Column(DateTime)

    user = relationship("User", back_populates="tax_returns")
    payments = relationship("Payment", back_populates="tax_return")

class Payment(Base):
    __tablename__ = "payments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    tax_return_id = Column(Integer, ForeignKey("tax_returns.id"))
    amount = Column(Float)
    payment_method = Column(String)
    status = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="payments")
    tax_return = relationship("TaxReturn", back_populates="payments")

# Pydantic Models
class UserCreate(BaseModel):
    email: EmailStr
    full_name: str
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    is_active: bool
    is_cpa: bool
    created_at: datetime

    class Config:
        from_attributes = True

class TaxReturnCreate(BaseModel):
    tax_year: int
    income: float
    deductions: Optional[float] = None
    withholdings: float = 0

class TaxReturnResponse(BaseModel):
    id: int
    tax_year: int
    income: float
    deductions: float
    withholdings: float
    tax_owed: float
    refund_amount: float
    amount_owed: float
    status: str
    created_at: datetime

    class Config:
        from_attributes = True

class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    uploaded_at: datetime
    processing_status: Optional[str] = None
    document_type: Optional[str] = None
    processed_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class PaymentCreate(BaseModel):
    tax_return_id: int
    amount: float

class PaymentResponse(BaseModel):
    id: int
    amount: float
    payment_method: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True

# Create tables
Base.metadata.create_all(bind=engine)

# FastAPI App
app = FastAPI(title="TaxBox.AI API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize document processor (if available)
try:
    from document_processor import DocumentProcessor
    doc_processor = DocumentProcessor()
except ImportError:
    doc_processor = None

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Auth functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    user = db.query(User).filter(User.email == username).first()
    if user is None:
        raise credentials_exception
    return user

# Routes
@app.get("/")
def root():
    return {"message": "TaxBox.AI API is running"}

@app.post("/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enhanced document upload with processing"""
    
    # Create uploads directory if it doesn't exist
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    
    # Generate unique filename
    file_extension = file.filename.split(".")[-1] if "." in file.filename else "unknown"
    unique_filename = f"{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    file_path = os.path.join(uploads_dir, unique_filename)
    
    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Create document record
    db_document = Document(
        user_id=current_user.id,
        filename=file.filename,
        file_path=file_path,
        file_type=file_extension,
        processing_status="pending"
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    
    # Process document if processor is available
    if doc_processor:
        try:
            # Update status to processing
            db_document.processing_status = "processing"
            db.commit()
            
            # Process the document
            result = doc_processor.process_document(file_path, file_extension)
            
            if result.get("success"):
                # Update document with extracted data
                db_document.processing_status = "completed"
                db_document.document_type = result.get("document_type")
                db_document.extracted_data = result.get("extracted_data")
                db_document.processed_at = datetime.utcnow()
                
                # Create suggestions
                suggestions = doc_processor.suggest_tax_entries(result.get("extracted_data", {}))
                for suggestion in suggestions:
                    db_suggestion = DocumentSuggestion(
                        document_id=db_document.id,
                        field_name=suggestion["field"],
                        suggested_value=suggestion.get("suggested_value"),
                        description=suggestion["description"],
                        confidence=suggestion["confidence"]
                    )
                    db.add(db_suggestion)
                    
            else:
                db_document.processing_status = "failed"
                db_document.processing_error = result.get("error")
                
            db.commit()
            
        except Exception as e:
            db_document.processing_status = "failed"
            db_document.processing_error = str(e)
            db.commit()
    else:
        # No processor available, mark as completed
        db_document.processing_status = "completed"
        db_document.document_type = "unknown"
        db.commit()
    
    return db_document

@app.get("/documents")
def get_documents(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(Document).filter(Document.user_id == current_user.id).all()

@app.post("/tax-returns", response_model=TaxReturnResponse)
def create_tax_return(
    tax_return: TaxReturnCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Tax calculation
    total_income = tax_return.income
    deductions = tax_return.deductions or 12550  # Standard deduction
    taxable_income = max(0, total_income - deductions)

    # Simplified tax calculation
    if taxable_income <= 10275:
        tax_owed = taxable_income * 0.10
    elif taxable_income <= 41775:
        tax_owed = 1027.50 + (taxable_income - 10275) * 0.12
    else:
        tax_owed = 4807.50 + (taxable_income - 41775) * 0.22

    refund_amount = max(0, tax_return.withholdings - tax_owed)
    amount_owed = max(0, tax_owed - tax_return.withholdings)

    db_tax_return = TaxReturn(
        user_id=current_user.id,
        tax_year=tax_return.tax_year,
        income=tax_return.income,
        deductions=deductions,
        withholdings=tax_return.withholdings,
        tax_owed=tax_owed,
        refund_amount=refund_amount,
        amount_owed=amount_owed,
        status="draft"
    )
    db.add(db_tax_return)
    db.commit()
    db.refresh(db_tax_return)
    return db_tax_return

@app.get("/tax-returns")
def get_tax_returns(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(TaxReturn).filter(TaxReturn.user_id == current_user.id).all()

@app.get("/tax-returns/{tax_return_id}/export/json")
def export_tax_return_json(
    tax_return_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Export tax return as JSON file"""
    
    # Get the tax return
    tax_return = db.query(TaxReturn).filter(
        TaxReturn.id == tax_return_id,
        TaxReturn.user_id == current_user.id
    ).first()
    
    if not tax_return:
        raise HTTPException(status_code=404, detail="Tax return not found")
    
    # Create comprehensive JSON data
    export_data = {
        "tax_summary": {
            "generated_at": datetime.now().isoformat(),
            "tax_year": tax_return.tax_year,
            "status": tax_return.status,
            "created_at": tax_return.created_at.isoformat(),
            "submitted_at": tax_return.submitted_at.isoformat() if tax_return.submitted_at else None
        },
        "taxpayer_info": {
            "name": current_user.full_name,
            "email": current_user.email,
            "filing_status": "Single"
        },
        "income_information": {
            "total_income": tax_return.income,
            "income_sources": [
                {
                    "source": "W-2 Wages",
                    "amount": tax_return.income
                }
            ]
        },
        "deductions": {
            "total_deductions": tax_return.deductions,
            "deduction_type": "Standard" if tax_return.deductions <= 12550 else "Itemized",
            "standard_deduction": 12550,
            "itemized_deductions": max(0, tax_return.deductions - 12550)
        },
        "tax_calculation": {
            "taxable_income": max(0, tax_return.income - tax_return.deductions),
            "tax_owed": tax_return.tax_owed,
            "withholdings": tax_return.withholdings,
            "refund_amount": tax_return.refund_amount,
            "amount_owed": tax_return.amount_owed
        },
        "payment_info": {
            "refund_due": tax_return.refund_amount > 0,
            "payment_required": tax_return.amount_owed > 0,
            "net_amount": tax_return.refund_amount - tax_return.amount_owed
        }
    }
    
    # Create response with JSON file
    return JSONResponse(
        content=export_data,
        headers={
            "Content-Disposition": f"attachment; filename=tax_return_{tax_return.tax_year}_{current_user.id}.json"
        }
    )

@app.post("/payments", response_model=PaymentResponse)
def create_payment(
    payment: PaymentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_payment = Payment(
        user_id=current_user.id,
        tax_return_id=payment.tax_return_id,
        amount=payment.amount,
        payment_method="credit_card",
        status="completed"
    )
    db.add(db_payment)
    db.commit()
    db.refresh(db_payment)
    return db_payment

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
