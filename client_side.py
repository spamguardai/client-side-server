from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from config import MODEL_PATH
import piheaan as heaan
import base64
import tempfile
from Client.encrypt_decrypt import (
    init_context_and_keys,
    preprocess_and_encrypt,
    encrypt_weights,
    decrypt_result
)

# ----------------------------- #
# ⚙️ Initialize FastAPI application
# ----------------------------- #
app = FastAPI()

# CORS Config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------- #
# 🔑 Initialize encryption context, secret key, encryptor, and encoder
# - These are required for all HE operations
# ---------------------------------------------------- #
context, sk, encryptor, encoder = init_context_and_keys()

# ---------------------------------------------------- #
# 📦 Request model: input format for client-side email text
# ---------------------------------------------------- #
class EncryptMailRequest(BaseModel):
    text: str  # Example: the email body text

# ------------------------------------------------------------------------- #
# 🔐 Encryption endpoint (/encrypt)
# - Receives plaintext, vectorizes and encrypts it
# - Also encrypts the model weights and returns everything in base64 format
# ------------------------------------------------------------------------- #
@app.post("/encrypt")
def run_model(request: EncryptMailRequest):
    # 1️⃣ Preprocess and encrypt the input text to get a list of ciphertext vectors
    encrypted_vec_list = preprocess_and_encrypt(request.text, context, sk, encryptor, encoder, log_slots=15)

    # 2️⃣ Convert encrypted vectors to base64 strings for safe transmission
    encrypted_base64_list = []
    for ciphertext in encrypted_vec_list:
        with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
            ciphertext.save(tmp_file.name)
            tmp_file.seek(0)
            b = tmp_file.read()
            b64 = base64.b64encode(b).decode("utf-8")
            encrypted_base64_list.append(b64)

    # 3️⃣ Load pre-trained model and extract weights
    model = joblib.load(MODEL_PATH)
    coef = model.coef_[0]       # Model weights (numpy array)
    intercept = model.intercept_[0]  # Intercept term (float)

    # 4️⃣ Encrypt the model weights and intercept
    encrypted_coef, encrypted_intercept = encrypt_weights(coef, intercept, context, sk, encryptor, encoder)

    # 5️⃣ Convert encrypted weights to base64
    encrypted_coef_b64_list = []
    for coef in encrypted_coef:
        with tempfile.NamedTemporaryFile(delete=True) as coef_file:
            coef.save(coef_file.name)
            coef_file.seek(0)
            coef_b64 = base64.b64encode(coef_file.read()).decode("utf-8")
            encrypted_coef_b64_list.append(coef_b64)

    # 6️⃣ Convert encrypted intercept to base64
    with tempfile.NamedTemporaryFile(delete=True) as intercept_file:
        encrypted_intercept.save(intercept_file.name)
        intercept_file.seek(0)
        intercept_b64 = base64.b64encode(intercept_file.read()).decode("utf-8")

    # 7️⃣ Return all encrypted components in base64 format
    return {
        "encrypted_vector": encrypted_base64_list,         # List[str]
        "encrypted_coef": encrypted_coef_b64_list,         # List[str]
        "encrypted_intercept": intercept_b64               # str
    }


# ---------------------------------------------------- #
# 📦 Decryption request model
# - Receives base64-encoded encrypted result
# ---------------------------------------------------- #
class DecryptRequest(BaseModel):
    encrypted_result: str  # Encrypted ciphertext in base64 string

# ---------------------------------------------------------------- #
# 🔁 Utility: Convert a base64 string back to a HEAAN Ciphertext
# ---------------------------------------------------------------- #
def recover_ciphertext(b64: str, context: heaan.Context) -> heaan.Ciphertext:
    ct = heaan.Ciphertext(context)
    binary_data = base64.b64decode(b64)  # Convert string → binary
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        tmp_file.write(binary_data)
        tmp_file.flush()
        ct.load(tmp_file.name)  # Load Ciphertext from file
    return ct

# ------------------------------------------------------------------------------ #
# 🔓 Decryption endpoint (/decrypt-result)
# - Accepts an encrypted result in base64 and returns the decrypted real value
# ------------------------------------------------------------------------------ #
@app.post("/decrypt-result")
def decrypt_api(request: DecryptRequest):
    try:
        # 1️⃣ Recover ciphertext object from base64 string
        ciphertext = recover_ciphertext(request.encrypted_result, context)
        
        # 2️⃣ Decrypt the result using secret key
        result = decrypt_result(ciphertext, context, sk)
        
        # 3️⃣ Extract real (and imaginary) values from the result
        real = result.real
        imag = result.imag  # Can be used if needed

        # 4️⃣ Return decrypted real value
        return {
            "decrypted_value": {
                "real": float(real)
            }
        }
    except Exception as e:
        # ❗Return error details in case of failure
        return {"status": "error", "message": str(e)}
