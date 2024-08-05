import secrets

from utils.KeyManager import KeyManager


class KeyManagerSimetric(KeyManager):
    def __init__(self, key_length=16):
        self.key_length = key_length
        self.key = None

    def generate_key(self):
        self.key = secrets.token_bytes(self.key_length)

    def load_key(self, key_data):
        self.key = key_data

    def get_key(self):
        if not self.key:
            self.generate_key()
        return self.key
