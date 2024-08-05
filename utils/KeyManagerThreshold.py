import tss

from utils.KeyManager import KeyManager


class KeyManagerThreshold(KeyManager):
    def __init__(self, num_shares=255, threshold=2, identifier=b'1234'):
        self.secret = None
        self.shares = None
        self.num_shares = num_shares
        self.threshold = threshold
        self.identifier = identifier

    def generate_key(self):
        if self.secret is None:
            raise ValueError("Secret must be set before generating shares.")
        # Convert the secret to a string if it's in bytes
        # secret_str = self.secret.decode('utf-8') if isinstance(self.secret, bytes) else self.secret

        self.shares = tss.share_secret(secret=self.secret, nshares=self.num_shares, threshold=self.threshold,
                                       identifier=self.identifier)

    def set_secret(self, secret):
        self.secret = secret

    def get_shares(self):
        return self.shares

    def reconstruct_key(self, shares):
        return tss.reconstruct_secret(shares, self.identifier)
