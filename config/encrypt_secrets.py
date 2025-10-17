import os
from cryptography.fernet import Fernet

# 경로 설정
KEY_FILE = "config/secret.key"
PLAIN_FILE = "config/secrets.yaml"
ENCRYPTED_FILE = "config/secrets.yaml.enc"

def generate_key():
    """새 암호화 키 생성"""
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    print("🔑 secret.key created!")

def encrypt_secrets():
    """평문 secrets.yaml → 암호화 파일로 변환"""
    if not os.path.exists(KEY_FILE):
        generate_key()

    with open(KEY_FILE, "rb") as f:
        key = f.read()
    cipher = Fernet(key)

    if not os.path.exists(PLAIN_FILE):
        print(f"❌ {PLAIN_FILE} not found.")
        return

    with open(PLAIN_FILE, "rb") as f:
        plain_data = f.read()

    encrypted_data = cipher.encrypt(plain_data)

    with open(ENCRYPTED_FILE, "wb") as f:
        f.write(encrypted_data)

    print(f"✅ Encrypted and saved to {ENCRYPTED_FILE}")
    os.remove(PLAIN_FILE)
    print("🧹 Deleted plain secrets.yaml for safety.")

if __name__ == "__main__":
    encrypt_secrets()
