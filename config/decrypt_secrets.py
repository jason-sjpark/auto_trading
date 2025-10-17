from cryptography.fernet import Fernet
import yaml

KEY_FILE = "config/secret.key"
ENCRYPTED_FILE = "config/secrets.yaml.enc"

def decrypt_yaml():
    """복호화된 secrets.yaml 내용을 dict로 반환"""
    with open(KEY_FILE, "rb") as f:
        key = f.read()
    cipher = Fernet(key)

    with open(ENCRYPTED_FILE, "rb") as f:
        encrypted_data = f.read()

    decrypted = cipher.decrypt(encrypted_data)
    secrets = yaml.safe_load(decrypted)
    return secrets

if __name__ == "__main__":
    secrets = decrypt_yaml()
    print("🔓 Decrypted secrets:")
    print(secrets)
