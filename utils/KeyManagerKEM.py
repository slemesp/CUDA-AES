import os
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Random import get_random_bytes
from utils.KeyManager import KeyManager

class KeyManagerKEM(KeyManager):
    def __init__(self, key_size=2048):
        self.key_size = key_size
        self.private_key = None
        self.public_key = None

    def generate_key(self):
        key = RSA.generate(self.key_size)
        self.private_key = key
        self.public_key = key.publickey()

    def get_private_key(self):
        if self.private_key is None:
            raise ValueError("Private key has not been generated yet.")
        return self.private_key.export_key()

    def get_public_key(self):
        if self.public_key is None:
            raise ValueError("Public key has not been generated yet.")
        return self.public_key.export_key()

    def encapsulate_key(self):
        if self.public_key is None:
            raise ValueError("Public key has not been generated yet.")
        cipher_rsa = PKCS1_OAEP.new(self.public_key)
        symmetric_key = get_random_bytes(16)
        encrypted_key = cipher_rsa.encrypt(symmetric_key)
        return encrypted_key, symmetric_key

    def decapsulate_key(self, encrypted_key):
        if self.private_key is None:
            raise ValueError("Private key has not been generated yet.")
        cipher_rsa = PKCS1_OAEP.new(self.private_key)
        symmetric_key = cipher_rsa.decrypt(encrypted_key)
        return symmetric_key


# Ejemplo de uso
if __name__ == "__main__":
    kem_key_manager = KeyManagerKEM()
    kem_key_manager.generate_key()

    public_key = kem_key_manager.get_public_key()
    private_key = kem_key_manager.get_private_key()

    # Encapsular clave simétrica usando la clave pública
    ciphertext, shared_secret_enc = kem_key_manager.encapsulate_key()

    # Desencapsular clave simétrica usando la clave privada
    shared_secret_dec = kem_key_manager.decapsulate_key(private_key)

    assert shared_secret_enc == shared_secret_dec
    print("Claves compartidas coinciden y son seguras.")
