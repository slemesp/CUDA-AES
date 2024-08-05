import logging
import random

import numpy as np
from astropy.io import fits
from pycuda import autoinit

from AES import AES
from utils.KeyManagerAsimetric import KeyManagerAsimetric
from utils.KeyManagerKEM import KeyManagerKEM
from utils.KeyManagerSimetric import KeyManagerSimetric
from utils.KeyManagerThreshold import KeyManagerThreshold
from utils.KeyUser import KeyUser

# Configuración del logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AESStats():
    def __init__(self, fits_file, key):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.info("************ AES %s ************", fits_file)
        # key = key
        self.fits_file = fits_file
        self.fits_data = None
        self.fits_header = None
        self.counter = 14111985

    def set_log_level(self, level):
        self.logger.setLevel(level)

    def get_data(self):
        if self.fits_data is None:
            start_time = time.time()
            self.fits_data = fits.getdata(self.fits_file)
            logger.info("Data loaded in %.2f seconds", time.time() - start_time)
            logger.info("Data shape: %s", self.fits_data.shape)
        return self.fits_data

    def get_header(self):
        if self.fits_header is None:
            self.fits_header = fits.getheader(self.fits_file).tostring()
        return self.fits_header

    def data_to_bytes(self):
        return self.get_data().tobytes()

    def header_to_bytes(self):
        return self.get_header().encode('utf-8')

    def frombuffer_data(self, data_bytes):
        return np.frombuffer(data_bytes, dtype=self.get_data().dtype).reshape(self.get_data().shape)

    def frombuffer_header(self, header_bytes):
        return np.frombuffer(header_bytes, dtype=np.byte)

    def compute_aes_ebs_data(self, key_encryption, key_decryption):
        self.logger.info("*** AES EBC ***")
        aes = AES()
        start_time = time.time()
        data_bytes = self.data_to_bytes()
        key_array = np.frombuffer(key_encryption, dtype=np.byte)
        start_time_encryption = time.time()
        encrypt_bytes = aes.encrypt_gpu(data_bytes, key_array)
        encrypted_data = self.frombuffer_data(encrypt_bytes)
        logger.info("Encryption complete in %.2f seconds", time.time() - start_time_encryption)

        key_array = np.frombuffer(key_decryption, dtype=np.byte)
        start_time_decryption = time.time()
        decrypt_bytes = aes.decrypt_gpu(encrypt_bytes, key_array)
        decrypted_data = self.frombuffer_data(decrypt_bytes)
        logger.info("Decryption complete in %.2f seconds", time.time() - start_time_decryption)

        logger.info("Total time: %.2f seconds", time.time() - start_time)
        return encrypted_data, decrypted_data

    def compute_aes_ctr_data(self, key_encryption, key_decryption):
        self.logger.info("*** AES CTR ***")
        aes = AES()
        start_time = time.time()
        data_bytes = self.data_to_bytes()
        key_array = np.frombuffer(key_encryption, dtype=np.byte)
        start_time_encryption = time.time()
        encrypt_bytes = aes.encrypt_ctr_gpu(data_bytes, key_array, self.counter)
        encrypted_data = self.frombuffer_data(encrypt_bytes)
        logger.info("Encryption complete in %.2f seconds", time.time() - start_time_encryption)

        key_array = np.frombuffer(key_decryption, dtype=np.byte)
        start_time_decryption = time.time()
        decrypt_bytes = aes.encrypt_ctr_gpu(encrypt_bytes, key_array, self.counter)
        decrypted_data = self.frombuffer_data(decrypt_bytes)
        logger.info("Decryption complete in %.2f seconds", time.time() - start_time_decryption)

        logger.info("Total time: %.2f seconds", time.time() - start_time)
        return encrypted_data, decrypted_data

    def compute_aes_ebs_header(self, key_encryption, key_decryption):
        self.logger.info("*** AES EBC ***")
        aes = AES()
        start_time = time.time()
        header_bytes = self.header_to_bytes()
        key_array = np.frombuffer(key_encryption, dtype=np.byte)
        start_time_encryption = time.time()
        encrypt_bytes = aes.encrypt_gpu(header_bytes, key_array)
        encrypt_bytes_hex = bytes(encrypt_bytes).hex()
        logger.info("Encryption complete in %.2f seconds", time.time() - start_time_encryption)

        key_array = np.frombuffer(key_decryption, dtype=np.byte)
        start_time_decryption = time.time()
        decrypt_bytes = aes.decrypt_gpu(encrypt_bytes, key_array)
        decrypt_bytes = "".join([chr(item) for item in decrypt_bytes])
        decrypt_bytes = decrypt_bytes[:len(self.get_header())]
        logger.info("Decryption complete in %.2f seconds", time.time() - start_time_decryption)

        logger.info("Total time: %.2f seconds", time.time() - start_time)
        return encrypt_bytes_hex, decrypt_bytes

    def compute_aes_ctr_header(self, key_encryption, key_decryption):
        self.logger.info("*** AES CTR ***")
        aes = AES()
        start_time = time.time()
        header_bytes = self.header_to_bytes()
        key_array = np.frombuffer(key_encryption, dtype=np.byte)
        start_time_encryption = time.time()
        encrypt_bytes = aes.encrypt_ctr_gpu(header_bytes, key_array, self.counter)
        encrypt_bytes_hex = bytes(encrypt_bytes).hex()
        logger.info("Encryption complete in %.2f seconds", time.time() - start_time_encryption)

        key_array = np.frombuffer(key_decryption, dtype=np.byte)
        start_time_decryption = time.time()
        decrypt_bytes = aes.encrypt_ctr_gpu(encrypt_bytes, key_array, self.counter)
        decrypt_bytes = "".join([chr(item) for item in decrypt_bytes])
        decrypt_bytes = decrypt_bytes[:len(self.get_header())]
        logger.info("Decryption complete in %.2f seconds", time.time() - start_time_decryption)

        logger.info("Total time: %.2f seconds", time.time() - start_time)
        return encrypt_bytes_hex, decrypt_bytes

    def analyze_entropy(self, data):
        aes = AES()
        entropy = aes.calculate_entropy(data)
        logger.info("Entropy: %s", entropy)

    def check_results_data(self, decrypted_data, encrypted_data):
        size_show = 4
        logger.info("Original data: %s", self.get_data().ravel()[:size_show])
        logger.info("Decrypted data: %s", decrypted_data.ravel()[:size_show])
        logger.info("Encrypted data: %s", encrypted_data.ravel()[:size_show])

        if np.array_equal(decrypted_data, encrypted_data):
            logger.error("#######    Decrypted data is equal to the encrypted data")
        else:
            if self.get_data().all() != decrypted_data.all():
                logger.error("#######    Decrypted data is different to the original data, Shapes: %s, %s",
                             decrypted_data.shape, self.get_data().shape)
            else:
                pass
                # logger.info("Decrypted data is different from the original data")

    def analyze_results_data(self, encrypted_data, decrypted_data):
        self.check_results_data(decrypted_data, encrypted_data)
        self.analyze_entropy(encrypted_data)
        self.analyze_entropy(decrypted_data)

    def analyze_aes(self):
        key_length = 2048
        logger.info("***************************** AES KeyManagerSimetric")

        # Uso de KeyManagerSimetric para cifrado y descifrado simétrico
        sim_key_manager = KeyManagerSimetric(key_length=key_length)
        sim_key_user = KeyUser(sim_key_manager)

        # Generar la clave simétrica
        sim_key = sim_key_user.get_key()

        logger.info(" AES Data Analysis")

        encrypted_data_ebs, decrypted_data_ebs = self.compute_aes_ebs_data(key_encryption=sim_key,
                                                                           key_decryption=sim_key)
        self.analyze_results_data(encrypted_data_ebs, decrypted_data_ebs)
        encrypted_data_ctr, decrypted_data_ctr = self.compute_aes_ctr_data(key_encryption=sim_key,
                                                                           key_decryption=sim_key)
        self.analyze_results_data(encrypted_data_ctr, decrypted_data_ctr)

        logger.info(" AES Header Analysis")

        encrypted_header_ebs, decrypted_header_ebs = self.compute_aes_ebs_header(key_encryption=sim_key,
                                                                                 key_decryption=sim_key)
        self.analyze_results_header(encrypted_header_ebs, decrypted_header_ebs)
        encrypted_header_ctr, decrypted_header_ctr = self.compute_aes_ctr_header(key_encryption=sim_key,
                                                                                 key_decryption=sim_key)
        self.analyze_results_header(encrypted_header_ctr, decrypted_header_ctr)

        logger.info("***************************** AES KeyManagerAsimetric")

        # Crear una instancia de KeyManagerAsimetric
        asym_key_manager = KeyManagerAsimetric()

        # Generar las claves asimétricas
        asym_key_manager.generate_key()

        # Cifrar la clave simétrica con la clave pública
        symmetric_key = os.urandom(16)
        encrypted_key = asym_key_manager.encrypt_symmetric_key(symmetric_key)
        decrypted_symmetric_key = asym_key_manager.decrypt_symmetric_key(encrypted_key)

        logger.info(" AES Data Analysis")

        encrypted_data_ebs, decrypted_data_ebs = self.compute_aes_ebs_data(key_encryption=symmetric_key,
                                                                           key_decryption=decrypted_symmetric_key)
        self.analyze_results_data(encrypted_data_ebs, decrypted_data_ebs)
        encrypted_data_ctr, decrypted_data_ctr = self.compute_aes_ctr_data(key_encryption=symmetric_key,
                                                                           key_decryption=decrypted_symmetric_key)
        self.analyze_results_data(encrypted_data_ctr, decrypted_data_ctr)

        logger.info(" AES Header Analysis")

        encrypted_header_ebs, decrypted_header_ebs = self.compute_aes_ebs_header(key_encryption=decrypted_symmetric_key,
                                                                                 key_decryption=symmetric_key)
        self.analyze_results_header(encrypted_header_ebs, decrypted_header_ebs)
        encrypted_header_ctr, decrypted_header_ctr = self.compute_aes_ctr_header(key_encryption=decrypted_symmetric_key,
                                                                                 key_decryption=symmetric_key)
        self.analyze_results_header(encrypted_header_ctr, decrypted_header_ctr)

        logger.info("***************************** AES KeyManagerThreshold")

        # Crear una instancia de KeyManagerThreshold
        threshold_key_manager = KeyManagerThreshold(num_shares=254, threshold=2)

        # Establecer el secreto (por ejemplo, una clave simétrica)
        secret_key = KeyUser(KeyManagerSimetric(key_length=key_length)).get_key()
        threshold_key_manager.set_secret(secret_key)

        # Generar las claves de umbral
        threshold_key_manager.generate_key()

        # Obtener las partes generadas
        shares = threshold_key_manager.get_shares()

        # Reconstruir la clave usando algunas de las partes (mínimo el umbral)
        reconstructed_key_1 = threshold_key_manager.reconstruct_key(random.choices(shares, k=2))

        size_reconstructed_key = random.randint(2, len(shares))
        reconstructed_key_2 = threshold_key_manager.reconstruct_key(random.choices(shares, k=size_reconstructed_key))

        logger.info(" AES Data Analysis with 2 Shares")

        encrypted_data_ebs, decrypted_data_ebs = self.compute_aes_ebs_data(key_encryption=secret_key,
                                                                           key_decryption=reconstructed_key_1)
        self.analyze_results_data(encrypted_data_ebs, decrypted_data_ebs)
        encrypted_data_ctr, decrypted_data_ctr = self.compute_aes_ctr_data(key_encryption=secret_key,
                                                                           key_decryption=reconstructed_key_1)
        self.analyze_results_data(encrypted_data_ctr, decrypted_data_ctr)

        logger.info(" AES Header Analysis with 2 Shares")

        encrypted_header_ebs, decrypted_header_ebs = self.compute_aes_ebs_header(key_encryption=secret_key,
                                                                                 key_decryption=reconstructed_key_1)
        self.analyze_results_header(encrypted_header_ebs, decrypted_header_ebs)
        encrypted_header_ctr, decrypted_header_ctr = self.compute_aes_ctr_header(key_encryption=secret_key,
                                                                                 key_decryption=reconstructed_key_1)
        self.analyze_results_header(encrypted_header_ctr, decrypted_header_ctr)

        logger.info(" AES Data Analysis with %d Shares", size_reconstructed_key)

        encrypted_data_ebs, decrypted_data_ebs = self.compute_aes_ebs_data(key_encryption=secret_key,
                                                                           key_decryption=reconstructed_key_2)
        self.analyze_results_data(encrypted_data_ebs, decrypted_data_ebs)
        encrypted_data_ctr, decrypted_data_ctr = self.compute_aes_ctr_data(key_encryption=secret_key,
                                                                           key_decryption=reconstructed_key_2)
        self.analyze_results_data(encrypted_data_ctr, decrypted_data_ctr)

        logger.info(" AES Header Analysis with %d Shares", size_reconstructed_key)

        encrypted_header_ebs, decrypted_header_ebs = self.compute_aes_ebs_header(key_encryption=secret_key,
                                                                                 key_decryption=reconstructed_key_2)
        self.analyze_results_header(encrypted_header_ebs, decrypted_header_ebs)
        encrypted_header_ctr, decrypted_header_ctr = self.compute_aes_ctr_header(key_encryption=secret_key,
                                                                                 key_decryption=reconstructed_key_2)
        self.analyze_results_header(encrypted_header_ctr, decrypted_header_ctr)

        logger.info("***************************** AES KeyManagerKEM")

        # Crear una instancia de KeyManagerKEM
        kem_key_manager = KeyManagerKEM()

        # Generar las claves KEM
        kem_key_manager.generate_key()

        # Crear usuario con el gestor de claves KEM
        kem_key_user = KeyUser(kem_key_manager)

        # Encapsular una clave simétrica con la clave pública
        ciphertext, symmetric_key = kem_key_manager.encapsulate_key()
        decrypted_symmetric_key = kem_key_manager.decapsulate_key(ciphertext)

        logger.info(" AES Data Analysis")

        encrypted_data_ebs, decrypted_data_ebs = self.compute_aes_ebs_data(key_encryption=symmetric_key,
                                                                           key_decryption=decrypted_symmetric_key)
        self.analyze_results_data(encrypted_data_ebs, decrypted_data_ebs)
        encrypted_data_ctr, decrypted_data_ctr = self.compute_aes_ctr_data(key_encryption=symmetric_key,
                                                                           key_decryption=decrypted_symmetric_key)
        self.analyze_results_data(encrypted_data_ctr, decrypted_data_ctr)

        logger.info(" AES Header Analysis")

        encrypted_header_ebs, decrypted_header_ebs = self.compute_aes_ebs_header(key_encryption=symmetric_key,
                                                                                 key_decryption=decrypted_symmetric_key)
        self.analyze_results_header(encrypted_header_ebs, decrypted_header_ebs)
        encrypted_header_ctr, decrypted_header_ctr = self.compute_aes_ctr_header(key_encryption=symmetric_key,
                                                                                 key_decryption=decrypted_symmetric_key)
        self.analyze_results_header(encrypted_header_ctr, decrypted_header_ctr)

        logger.info("************ END %s ************", self.fits_file)

    def analyze_results_header(self, encrypted_header_ebs, decrypted_header_ebs):
        self.check_results_header(decrypted_header_ebs, encrypted_header_ebs)

    def check_results_header(self, decrypted_header_ebs, encrypted_header_ebs):
        size_show = 50
        logger.info("Original header: %s", self.get_header()[:size_show])
        logger.info("Decrypted header: %s", decrypted_header_ebs[:size_show])
        logger.info("Encrypted header: %s", encrypted_header_ebs[:size_show])

        # Comparación de arrays usando numpy.array_equal
        if np.array_equal(decrypted_header_ebs, encrypted_header_ebs):
            logger.error("#######    Decrypted header is equal to the encrypted header")
        else:
            if not np.array_equal(self.get_header(), decrypted_header_ebs):
                logger.error("#######    Decrypted header is different to the original header")

                # Comparar las longitudes
                if len(decrypted_header_ebs) != len(self.get_header()):
                    logger.error(
                        f"Longitud diferente: decrypted_header_ebs({len(decrypted_header_ebs)}) != original_header({len(self.get_header())})")
                else:
                    # Identificar y reportar las diferencias en contenido
                    diferencias = np.where(decrypted_header_ebs != self.get_header())
                    logger.error(f"Diferencias en índices: {diferencias[0]}")
                    for idx in diferencias[0]:
                        logger.error(
                            f"Diferencia en índice {idx}: decrypted_header_ebs={decrypted_header_ebs[idx]}, original_header={self.get_header()[idx]}")
            else:
                pass
                # logger.info("Decrypted header is different from the original header")


if __name__ == "__main__":
    import logging
    import time

    # Configuración del logger
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Autoinit device: %s", autoinit.device)

    import time
    import traceback
    import os

    # Root image directory
    directory_path = os.path.join(os.path.dirname(__file__), 'Images')
    directory_path = os.path.join('/home/slemes/PycharmProjects/GPUPhotFinal/tests/data')

    # Get all subdirectories and find if exists a FITS file
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.fits'):
                if "TTT1_iKon936-1_MasterFlat_SDSSg_Bin11.fits" in file:
                    image_path = os.path.join(root, file)
                    try:
                        logger.info(f"Processing {image_path}")
                        start_time = time.time()

                        testing_key = b'pruebaKeyMaster'

                        aes_stats = AESStats(image_path, testing_key)
                        # aes_stats.set_log_level(logging.WARNING)
                        aes_stats.analyze_aes()

                        end_time = time.time()
                        logger.info(f"Elapsed time for {file}: {end_time - start_time:.2f} seconds")
                    except Exception as e:
                        logger.error(f"Error processing {image_path}: {e}")
                        logger.error("Traceback:")
                        traceback.print_exc()
                break

# clave secreta -> Datos de la imagen
# esquema umbral 2:n (n variable?) (sistemas de ecuaciones) -> Header imagen
"""
Tiempos de generar en distintos tipos
Medir entropía de la imagen


AES
Generación claves aes
esquema umbral para proteger claves


Generación de claves AES
Esquema kem 
Crystals KEM postcuantica  kyber
"""
