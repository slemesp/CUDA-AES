import secrets
from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

class User:
    def __init__(self, id):
        self.id = id
        self.share = None

def generate_master_secret():
    return secrets.token_bytes(32)  # 256-bit secret

def split_secret(secret, n, t):
    shares = [secrets.token_bytes(32) for _ in range(n)]
    return shares

def reconstruct_secret(shares):
    # In a real implementation, this would use Shamir's Secret Sharing
    # For simplicity, we're just XORing the shares
    return bytes(a ^ b for a, b in zip(shares[0], shares[1]))

def generate_symmetric_key():
    return get_random_bytes(32)

def encrypt(key, data):
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher.encrypt(pad(data.encode(), AES.block_size))

def decrypt(key, data):
    cipher = AES.new(key, AES.MODE_ECB)
    return unpad(cipher.decrypt(data), AES.block_size).decode()

def encrypt_with_master_public_key(public_key, data):
    cipher = PKCS1_OAEP.new(public_key)
    return cipher.encrypt(data)

def decrypt_with_master_private_key(private_key, data):
    cipher = PKCS1_OAEP.new(private_key)
    return cipher.decrypt(data)

def redistribute_shares(shares, new_n, t):
    # In a real implementation, this would redistribute the shares
    # For simplicity, we're just generating new random shares
    return shares + [secrets.token_bytes(32) for _ in range(new_n - len(shares))]

# Main execution
if __name__ == "__main__":
    # Generate master secret and RSA key pair
    S = generate_master_secret()
    key = RSA.generate(2048)
    private_key = key
    public_key = key.publickey()

    # Create users
    admin = User("admin")
    users = [User(f"user_{i}") for i in range(1, 11)]
    all_users = [admin] + users

    # Distribute shares
    shares = split_secret(S, len(all_users), 2)
    for user, share in zip(all_users, shares):
        user.share = share

    print("Shares distributed to all users.")

    # Create and encrypt a file for user 1
    file_key = generate_symmetric_key()
    file_content = "Contenido del archivo del usuario 1"
    encrypted_file = encrypt(file_key, file_content)
    encrypted_file_key = encrypt_with_master_public_key(public_key, file_key)

    print(f"File created and encrypted for {users[0].id}")

    # User 1 accesses their file
    reconstructed_S = reconstruct_secret([users[0].share, admin.share])
    decrypted_file_key = decrypt_with_master_private_key(private_key, encrypted_file_key)
    decrypted_file = decrypt(decrypted_file_key, encrypted_file)

    print(f"{users[0].id} accessed their file. Content: {decrypted_file}")

    # User 1 shares with User 2
    print(f"{users[0].id} shares their file with {users[1].id}")

    # User 2 accesses the shared file
    reconstructed_S = reconstruct_secret([users[0].share, admin.share])
    decrypted_file_key = decrypt_with_master_private_key(private_key, encrypted_file_key)
    decrypted_file = decrypt(decrypted_file_key, encrypted_file)

    print(f"{users[1].id} accessed the shared file. Content: {decrypted_file}")

    # Add new users
    new_user1 = User("new_user_1")
    new_user2 = User("new_user_2")
    all_users.extend([new_user1, new_user2])

    new_shares = redistribute_shares(shares, len(all_users), 2)
    for user, share in zip(all_users, new_shares):
        user.share = share

    print("New users added and shares redistributed.")

    # New user accesses a shared file
    reconstructed_S = reconstruct_secret([new_user1.share, admin.share])
    decrypted_file_key = decrypt_with_master_private_key(private_key, encrypted_file_key)
    decrypted_file = decrypt(decrypted_file_key, encrypted_file)

    print(f"{new_user1.id} accessed the shared file. Content: {decrypted_file}")