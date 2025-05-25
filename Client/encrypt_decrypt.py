import joblib
import numpy as np
import os
import piheaan as heaan
from preprocessing import preprocess, vectorize
from config import VECTORIZER_PATH, KEY_PATH, LOG_SLOTS, IMPORTANT_INDICES_PATH

# Initializes homomorphic encryption context and generates keys.
def init_context_and_keys():
    os.makedirs(KEY_PATH, exist_ok=True)
    params = heaan.ParameterPreset.FGb
    context = heaan.make_context(params)

    sk = heaan.SecretKey(context)
    # sk.save(os.path.join(KEY_PATH, "secret.key"))
    keygen = heaan.KeyGenerator(context, sk)
    keygen.gen_common_keys()
    keygen.gen_enc_key()
    keygen.save(KEY_PATH)

    encryptor = heaan.Encryptor(context)
    encoder = heaan.EnDecoder(context)

    return context, sk, encryptor, encoder

# Encrypts a numeric vector into ciphertexts.
def encrypt_vector(vec, context, sk, encryptor, encoder):
    encrypted_vec = []
    for val in vec:
        msg = heaan.Message(LOG_SLOTS)
        msg[0] = float(val)
        encoded = encoder.encode(msg, LOG_SLOTS)
        ctxt = heaan.Ciphertext(context)
        encryptor.encrypt(encoded, sk, ctxt)
        encrypted_vec.append(ctxt)
    return encrypted_vec

# Preprocesses text, vectorizes it, and encrypts the vector.
def preprocess_and_encrypt(user_text, context, sk, encryptor, encoder, log_slots=15):
    user_text = preprocess(user_text)
    vectorizer = joblib.load(VECTORIZER_PATH)
    important_indices = joblib.load(IMPORTANT_INDICES_PATH)
    X_vec, _ = vectorize([user_text], vectorizer, important_indices, fit_vectorizer=False)
    # print("Vectorized feature vector:", len(X_vec[0]))
    encrypted_vec = encrypt_vector(X_vec[0], context, sk, encryptor, encoder)
    return encrypted_vec

# Encrypts model coefficients and intercept.
def encrypt_weights(coef, intercept, context, sk, encryptor, encoder):
    encrypted_coef = []
    for val in coef:
        msg = heaan.Message(LOG_SLOTS)
        msg[0] = float(val)
        encoded = encoder.encode(msg, LOG_SLOTS)
        ctxt = heaan.Ciphertext(context)
        encryptor.encrypt(encoded, sk, ctxt)
        encrypted_coef.append(ctxt)
    # Encrypt intercept
    msg = heaan.Message(LOG_SLOTS)
    msg[0] = float(intercept)
    encoded = encoder.encode(msg, LOG_SLOTS)
    intercept_ctxt = heaan.Ciphertext(context)
    encryptor.encrypt(encoded, sk, intercept_ctxt)

    return encrypted_coef, intercept_ctxt

# Decrypts a ciphertext to obtain the original numeric result.
def decrypt_result(ciphertext, context, sk):
    decryptor = heaan.Decryptor(context)
    msg = heaan.Message(LOG_SLOTS)
    decryptor.decrypt(ciphertext, sk, msg)
    return msg[0]

# def main(user_input):
#     context, sk, encryptor, encoder = init_context_and_keys()
#     # Simple test: encrypt -> decrypt
#     encrypted_vec = preprocess_and_encrypt(user_input, context, sk, encryptor, encoder)
#     # Decrypt the first ciphertext for testing
#     decrypted_val = decrypt_result(encrypted_vec[0], context, sk)
#     print(f"Decrypted test value: {decrypted_val}")

# if __name__ == "__main__":
#     user_input = "hope enjoyed new content text stop unsubscribe helpp provided tonesyoucouk"
#     main(user_input)
