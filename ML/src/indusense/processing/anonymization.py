import hashlib

SALT = "incidents_v1"


def anon(name: str) -> str:
    h = hashlib.sha256(f"{SALT}:{name}".encode()).hexdigest()
    return f"OP_ANON_{h[:8].upper()}"
