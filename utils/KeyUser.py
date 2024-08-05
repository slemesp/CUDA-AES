from utils.KeyManagerAsimetric import KeyManagerAsimetric


class KeyUser:
    def __init__(self, key_manager):
        self.key_manager = key_manager

    def get_key(self):
        return self.key_manager.get_key()

    def get_public_key(self):
        if isinstance(self.key_manager, KeyManagerAsimetric):
            return self.key_manager.get_public_key()
        return None

    def get_private_key(self):
        if isinstance(self.key_manager, KeyManagerAsimetric):
            return self.key_manager.get_private_key()
        return None
