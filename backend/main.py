import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel, EmailStr
import jwt
from passlib.context import CryptContext
import uvicorn
import shutil
from enum import Enum

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./taxbox.db")

if "postgresql" in DATABASE_URL:
    connect_args = {"sslmode": "require"}
elif "sqlite" in DATABASE_URL:
    connect_args = {"check_same_thread": False}
else:
    connect_args = {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
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

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String)
    file_path = Column(String)
    file_type = Column(String)
    file_size = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processing_status = Column(String, default="completed")
    ocr_text = Column(Text, default="Sample text extracted")

class TaxReturn(Base):
    __tablename__ = "tax_returns"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    tax_year = Column(Integer)
    filing_status = Column(String, default="single")
    spouse_name = Column(String)
    spouse_ssn = Column(String)
    spouse_has_income = Column(Boolean, default=False)
    spouse_itemizes = Column(Boolean, default=False)
    qualifying_person_name = Column(String)
    qualifying_person_relationship = Column(String)
    lived_with_taxpayer = Column(Boolean, default=False)
    income = Column(Float)
    deductions = Column(Float)
    withholdings = Column(Float)
    tax_owed = Column(Float)
    refund_amount = Column(Float)
    amount_owed = Column(Float)
    status = Column(String, default="draft")
    created_at = Column(DateTime, default=datetime.utcnow)
    submitted_at = Column(DateTime)

class Payment(Base):
    __tablename__ = "payments"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    tax_return_id = Column(Integer, ForeignKey("tax_returns.id"))
    amount = Column(Float)
    payment_method = Column(String)
    status = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

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

class FilingStatus(str, Enum):
    SINGLE = "single"
    MARRIED_JOINTLY = "married_jointly"
    MARRIED_SEPARATELY = "married_separately"
    HEAD_OF_HOUSEHOLD = "head_of_household"
    QUALIFYING_WIDOW = "qualifying_widow"

class FilingStatusInfo(BaseModel):
    filing_status: FilingStatus
    spouse_name: Optional[str] = None
    spouse_ssn: Optional[str] = None
    spouse_has_income: Optional[bool] = False
    spouse_itemizes: Optional[bool] = False
    qualifying_person_name: Optional[str] = None
    qualifying_person_relationship: Optional[str] = None
    lived_with_taxpayer: Optional[bool] = False

class TaxReturnCreate(BaseModel):
    tax_year: int
    income: float
    deductions: Optional[float] = None
    withholdings: float = 0
    filing_status_info: Optional[FilingStatusInfo] = None

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
    filing_status: str
    spouse_name: Optional[str] = None
    spouse_ssn: Optional[str] = None
    spouse_has_income: Optional[bool] = None
    spouse_itemizes: Optional[bool] = None
    qualifying_person_name: Optional[str] = None
    qualifying_person_relationship: Optional[str] = None
    lived_with_taxpayer: Optional[bool] = None
    class Config:
        from_attributes = True

class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: Optional[int] = 0
    uploaded_at: datetime
    processing_status: Optional[str] = None
    ocr_text: Optional[str] = None
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

# FastAPI App
app = FastAPI(title="TaxBox.AI API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "TaxBox2025SuperSecretProductionKey32Chars")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

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

# Tax calculation functions
def get_standard_deduction(filing_status: str, tax_year: int = 2024):
    standard_deductions = {
        "single": 14600,
        "married_jointly": 29200,
        "married_separately": 14600,
        "head_of_household": 21900,
        "qualifying_widow": 29200
    }
    return standard_deductions.get(filing_status, 14600)

def calculate_tax_owed(taxable_income: float, filing_status: str):
    if filing_status in ["single", "married_separately"]:
        if taxable_income <= 11000:
            return taxable_income * 0.10
        elif taxable_income <= 44725:
            return 1100 + (taxable_income - 11000) * 0.12
        elif taxable_income <= 95375:
            return 5147 + (taxable_income - 44725) * 0.22
        else:
            return 16290 + (taxable_income - 95375) * 0.24
    elif filing_status in ["married_jointly", "qualifying_widow"]:
        if taxable_income <= 22000:
            return taxable_income * 0.10
        elif taxable_income <= 89450:
            return 2200 + (taxable_income - 22000) * 0.12
        elif taxable_income <= 190750:
            return 10294 + (taxable_income - 89450) * 0.22
        else:
            return 32580 + (taxable_income - 190750) * 0.24
    elif filing_status == "head_of_household":
        if taxable_income <= 15700:
            return taxable_income * 0.10
        elif taxable_income <= 59850:
            return 1570 + (taxable_income - 15700) * 0.12
        elif taxable_income <= 95350:
            return 6868 + (taxable_income - 59850) * 0.22
        else:
            return 14678 + (taxable_income - 95350) * 0.24
    return 0

# Routes
@app.get("/")
async def root():
    return {"message": "TaxBox.AI API is running", "status": "healthy"}

@app.post("/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    try:
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
    except Exception as e:
        print(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    try:
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
    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.get("/api/users/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/api/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        os.makedirs("uploads", exist_ok=True)
        file_extension = file.filename.split(".")[-1] if "." in file.filename else "unknown"
        unique_filename = f"{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        file_path = os.path.join("uploads", unique_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = os.path.getsize(file_path)
        
        db_document = Document(
            user_id=current_user.id,
            filename=file.filename,
            file_path=file_path,
            file_type=file_extension,
            file_size=file_size,
            processing_status="completed",
            ocr_text="Sample extracted text for demo purposes"
        )
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        
        return db_document
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")

@app.get("/api/documents", response_model=List[DocumentResponse])
def get_documents(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(Document).filter(Document.user_id == current_user.id).all()

@app.delete("/api/documents/{document_id}")
def delete_document(document_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == document_id, Document.user_id == current_user.id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document.file_path and os.path.exists(document.file_path):
        try:
            os.remove(document.file_path)
        except:
            pass
    
    db.delete(document)
    db.commit()
    return {"message": "Document deleted successfully"}

@app.post("/api/tax-returns", response_model=TaxReturnResponse)
def create_tax_return(tax_return: TaxReturnCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        filing_info = tax_return.filing_status_info
        filing_status = filing_info.filing_status if filing_info else FilingStatus.SINGLE
        
        total_income = tax_return.income
        deductions = tax_return.deductions or get_standard_deduction(filing_status, tax_return.tax_year)
        taxable_income = max(0, total_income - deductions)
        tax_owed = calculate_tax_owed(taxable_income, filing_status)
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
            status="draft",
            filing_status=filing_status,
            spouse_name=filing_info.spouse_name if filing_info else None,
            spouse_ssn=filing_info.spouse_ssn if filing_info else None,
            spouse_has_income=filing_info.spouse_has_income if filing_info else False,
            spouse_itemizes=filing_info.spouse_itemizes if filing_info else False,
            qualifying_person_name=filing_info.qualifying_person_name if filing_info else None,
            qualifying_person_relationship=filing_info.qualifying_person_relationship if filing_info else None,
            lived_with_taxpayer=filing_info.lived_with_taxpayer if filing_info else False
        )
        
        db.add(db_tax_return)
        db.commit()
        db.refresh(db_tax_return)
        return db_tax_return
    except Exception as e:
        print(f"Tax return creation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create tax return")

@app.get("/api/tax-returns", response_model=List[TaxReturnResponse])
def get_tax_returns(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(TaxReturn).filter(TaxReturn.user_id == current_user.id).all()

@app.get("/api/filing-status/standard-deductions")
def get_standard_deductions(tax_year: int = 2024):
    return {
        "tax_year": tax_year,
        "standard_deductions": {
            "single": get_standard_deduction("single", tax_year),
            "married_jointly": get_standard_deduction("married_jointly", tax_year),
            "married_separately": get_standard_deduction("married_separately", tax_year),
            "head_of_household": get_standard_deduction("head_of_household", tax_year),
            "qualifying_widow": get_standard_deduction("qualifying_widow", tax_year)
        }
    }

@app.get("/api/filing-status/options")
def get_filing_status_options():
    return {
        "filing_statuses": [
            {"value": "single", "label": "Single", "description": "Check if you are unmarried or legally separated"},
            {"value": "married_jointly", "label": "Married Filing Jointly", "description": "Check if you are married and file jointly"},
            {"value": "married_separately", "label": "Married Filing Separately", "description": "Check if you are married but file separately"},
            {"value": "head_of_household", "label": "Head of Household", "description": "Check if you are unmarried and paid more than half home costs"},
            {"value": "qualifying_widow", "label": "Qualifying Widow(er)", "description": "Check if your spouse died and you have a qualifying child"}
        ]
    }

@app.post("/api/payments", response_model=PaymentResponse)
def create_payment(payment: PaymentCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
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
