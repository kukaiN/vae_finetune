from safetensors.torch import load_file, save_file, safe_open
import torch
import os
import hashlib
import io
import random
import torch.nn.functional as F

from convert_to_SD import convert_vae_state_dict

output_dir = r"G:\models\SD_webui\VAE"
path_1 = os.path.join(output_dir, "sdxl_vae.safetensors")
path_2 = os.path.join(output_dir, "diffusion_pytorch_model_converted8_29.safetensors")
#path_2 = os.path.join(output_dir, "diffusion_pytorch_model_translated3.safetensors")



def print_metadata(path):
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata()
    print(metadata)

print_metadata(path_1)
print_metadata(path_2)

vae_1 = load_file(path_1)
#fp16_weights = {k: v.to(torch.float16) for k, v in loaded_weights.items()}
# Load the saved weights from the safetensors file
vae_2 = load_file(path_2)

vae_2 = convert_vae_state_dict(vae_2)

vae_1 = {k: v.to(torch.float16) for k, v in vae_1.items()}
vae_2 = {k: v.to(torch.float16) for k, v in vae_2.items()}

def calculate_sha256(filename):
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)
    # [0:10] is how they crop hashes in SD
    return hash_sha256.hexdigest()[0:10]
def sha256(filename, title=None, use_addnet_hash=False):
    #hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")

    #sha256_value = sha256_from_cache(filename, title, use_addnet_hash)
    #if sha256_value is not None:
    #    return sha256_value

    #if shared.cmd_opts.no_hashing:
    #    return None

    print(f"Calculating sha256 for {filename}: ", end='')
    #if use_addnet_hash:
    #    with open(filename, "rb") as file:
    #        sha256_value = addnet_hash_safetensors(file)
    #else:
    sha256_value = calculate_sha256(filename)
    print(f"{sha256_value}")

    #hashes[title] = {
    #    "mtime": os.path.getmtime(filename),
    #    "sha256": sha256_value,
    #}

    #dump_cache()
    return sha256_value

def get_hashes(model):
    buffer = io.BytesIO()
    # returned file object, but we don't use it
    buffer1 = torch.save(model, buffer, _use_new_zipfile_serialization=False)
    
    buffer.seek(0)  # Move to the beginning of the buffer
    hash_sha256 = hashlib.sha256(buffer.read()).hexdigest()
    buffer.seek(0)
    hash_md5 = hashlib.md5(buffer.read()).hexdigest()

    return hash_sha256[0:10], hash_md5[0:10]

#key_to_modify = random.choice(list(vae_2.keys()))
#vae_2[key_to_modify] += torch.randn_like(vae_2[key_to_modify]) + 1e-7

hash_sha256_1, hash_md5_1 = get_hashes(vae_1)
hash_sha256_2, hash_md5_2 = get_hashes(vae_2)

sha256_1 = sha256(path_1, 'vae')
sha256_2 = sha256(path_2, 'vae')

# Compare the saved weights with the current model's weights
for key in vae_1:
    if key in vae_2:
        weight_1 = vae_1[key]
        weight_2 = vae_2[key].to(torch.float16)
        if not torch.equal(weight_1, weight_2):
            mean_abs_diff = torch.mean(torch.abs(weight_1 - weight_2)).item()
            cosine_sim = F.cosine_similarity(weight_1.flatten(), weight_2.flatten(), dim=0).item()
            
            print(f"Weight mismatch found in layer {key}")
            print(f" - Mean Absolute Difference: {mean_abs_diff}")
            print(f" - Cosine Similarity: {cosine_sim}\n")
        else:
            print(f"Layer {key} matches.")
    else:
        print(f"Key {key} not found in saved weights.")
        
print(f"SHA256 HASH: {hash_sha256_1}, {hash_sha256_2}")
print(f"md5 hash: {hash_md5_1}, {hash_md5_2}")   
print(f"SDXL's SHA256: {sha256_1}, {sha256_2}")     


path_3 =os.path.join(output_dir, "diffusion_pytorch_model_converted8_29.safetensors")

save_file(vae_2, path_3, None)