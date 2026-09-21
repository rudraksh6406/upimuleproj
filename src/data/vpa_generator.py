"""Synthetic UPI VPA, Account, and Device Generator."""
import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple

FIRST_NAMES = [
    "aarav", "vihaan", "aditya", "arjun", "sai", "rohit", "rahul", "priya", 
    "sneha", "pooja", "ananya", "neha", "vikram", "amit", "deepak", "karan",
    "manoj", "suresh", "divya", "kavita", "swati", "meera", "rajesh", "sanjay",
    "alok", "varun", "nikhil", "ishaan", "tanvi", "riya", "shreya", "anita"
]

LAST_NAMES = [
    "sharma", "verma", "gupta", "singh", "kumar", "patel", "reddy", "nair",
    "iyer", "joshi", "rao", "deshmukh", "mehta", "chatterjee", "banerjee",
    "mukherjee", "das", "agarwal", "bhat", "kulkarni", "mishra", "dubey"
]

MERCHANTS = [
    "flipkart", "amazon", "swiggy", "zomato", "blinkit", "zepto", "dmart",
    "reliance_fresh", "uber", "ola", "airtel_bill", "jio_recharge", "tata_neu",
    "myntra", "nykaa", "bookmyshow", "irctc", "makemytrip", "local_kirana"
]

MERCHANT_CATEGORIES = [
    ("groceries", 0.25),
    ("food", 0.20),
    ("transport", 0.15),
    ("bills", 0.15),
    ("shopping", 0.10),
    ("entertainment", 0.10),
    ("other", 0.05),
]

INDIAN_CITIES = [
    ("Bengaluru", "Karnataka"),
    ("Mumbai", "Maharashtra"),
    ("Delhi", "Delhi"),
    ("Hyderabad", "Telangana"),
    ("Chennai", "Tamil Nadu"),
    ("Kolkata", "West Bengal"),
    ("Pune", "Maharashtra"),
    ("Ahmedabad", "Gujarat"),
    ("Jaipur", "Rajasthan"),
    ("Lucknow", "Uttar Pradesh"),
    ("Patna", "Bihar"),
    ("Bhopal", "Madhya Pradesh"),
    ("Chandigarh", "Punjab"),
]

UPI_APPS = ["GooglePay", "PhonePe", "Paytm", "BHIM", "Cred", "AmazonPay"]


def sample_bank_handle(bank_weights: Optional[List[Dict[str, Any]]] = None) -> str:
    """Samples a UPI bank handle based on weight distribution."""
    if not bank_weights:
        bank_weights = [
            {"name": "okhdfcbank", "weight": 0.25},
            {"name": "okaxis", "weight": 0.15},
            {"name": "okstatebank", "weight": 0.20},
            {"name": "okicici", "weight": 0.15},
            {"name": "okpunjab", "weight": 0.05},
            {"name": "ybl", "weight": 0.05},
            {"name": "okdbs", "weight": 0.03},
            {"name": "paytm", "weight": 0.12},
        ]
    names = [b["name"] for b in bank_weights]
    weights = [float(b["weight"]) for b in bank_weights]
    return random.choices(names, weights=weights, k=1)[0]


def generate_vpa(is_merchant: bool = False, bank_weights: Optional[List[Dict]] = None) -> str:
    """Generates a realistic Indian VPA."""
    bank = sample_bank_handle(bank_weights)
    if is_merchant:
        merchant = random.choice(MERCHANTS)
        suffix = random.randint(10, 9999) if random.random() < 0.4 else ""
        return f"merchant.{merchant}{suffix}@{bank}"
    else:
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        suffix = random.randint(1, 999) if random.random() < 0.7 else ""
        sep = "." if random.random() < 0.5 else ""
        return f"{first}{sep}{last}{suffix}@{bank}"


def generate_device_id(seed_str: Optional[str] = None) -> str:
    """Generates a synthetic device fingerprint hash."""
    raw = seed_str or f"device_{random.random()}_{random.randint(10000, 99999)}"
    return f"dev_{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


def generate_ip_address(city_name: Optional[str] = None) -> str:
    """Generates a synthetic Indian IP address."""
    subnets = {
        "Bengaluru": "106.51.",
        "Mumbai": "115.111.",
        "Delhi": "122.161.",
        "Hyderabad": "183.82.",
        "Jaipur": "49.36.",
    }
    prefix = subnets.get(city_name or "", f"{random.choice([49, 106, 115, 122, 183])}.{random.randint(10, 250)}.")
    return f"{prefix}{random.randint(1, 254)}.xxx"


def sample_merchant_category() -> str:
    """Samples a merchant category based on distribution."""
    cats = [c[0] for c in MERCHANT_CATEGORIES]
    weights = [c[1] for c in MERCHANT_CATEGORIES]
    return random.choices(cats, weights=weights, k=1)[0]
