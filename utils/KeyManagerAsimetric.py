from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from utils.KeyManager import KeyManager

class KeyManagerAsimetric(KeyManager):
    def __init__(self, key_size=2048, public_exponent=65537):
        self.private_key = None
        self.public_key = None
        self.key_size = key_size
        self.public_exponent = public_exponent

    def generate_key(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=self.public_exponent,
            key_size=self.key_size
        )
        self.public_key = self.private_key.public_key()

    def get_private_key(self):
        if self.private_key is None:
            raise ValueError("Private key has not been generated yet.")
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        )

    def get_public_key(self):
        if self.public_key is None:
            raise ValueError("Public key has not been generated yet.")
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

    def encrypt_symmetric_key(self, symmetric_key):
        if self.public_key is None:
            raise ValueError("Public key has not been generated yet.")
        encrypted_key = self.public_key.encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return encrypted_key

    def decrypt_symmetric_key(self, encrypted_key):
        if self.private_key is None:
            raise ValueError("Private key has not been generated yet.")
        symmetric_key = self.private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return symmetric_key
