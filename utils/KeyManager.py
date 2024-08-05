import os
import secrets
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key, load_pem_public_key, Encoding, PrivateFormat, PublicFormat, NoEncryption
)
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

class KeyManager:
    def generate_key(self):
        raise NotImplementedError

    def load_key(self, key_data):
        raise NotImplementedError

    def get_key(self):
        raise NotImplementedError
