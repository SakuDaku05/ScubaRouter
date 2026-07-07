import os
gguf = r"C:\Users\seema\.lmstudio\models\lmstudio-community\gemma-4-E4B-it-GGUF\gemma-4-E4B-it-Q4_K_M.gguf"
with open(gguf, 'rb') as f:
    magic   = f.read(4)
    version = int.from_bytes(f.read(4), 'little')
size_gb = os.path.getsize(gguf) / 1e9
print(f"Magic:        {magic}")
print(f"GGUF version: {version}")
print(f"File size:    {size_gb:.2f} GB")
